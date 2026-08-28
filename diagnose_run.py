"""Runs the full 200-session public set through starter.agent.Agent,
prints diagnostic tables, and then a VERDICTS summary of PASS/WARN/FAIL
checks against explicit thresholds.

Adapted from an earlier version built around src.dialog.DialogController
(Balance-scored question selection, a 0.0-1.0 penalty scale). That
architecture no longer exists in starter/agent.py, so the penalty-spread
and override-preparation sections were replaced with checks that match
this agent's actual design: a confidence gate that deliberately returns
fewer than 10 recommendations early, and a "rescue mode" that delays
Person C's heavier pipeline until the base agent stalls at turn 5.

Run: python3 diagnose_run.py
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10
RESCUE_AT = 5  # starter.agent's default C_RESCUE_TURN


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)


def print_table(headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    agent = Agent(CATALOG_PATH)

    recs_count_by_turn: dict[int, list[int]] = defaultdict(list)
    attribute_counts: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    first_no_pref_turn_by_scenario: dict[str, list[int]] = defaultdict(list)
    first_no_pref_none_count_by_scenario: Counter = Counter()
    rescue_outcomes: list[tuple[str, bool, int | None]] = []  # (scenario, reached_rescue, hit_turn)
    turn_durations: list[float] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        session_id = f"diag_{sample_id}"
        scenario_type = sample["scenario_type"]

        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario_type != "intent_override"
        user_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed
        )

        session_first_no_pref_turn: int | None = None
        hit_turn: int | None = None
        reached_rescue = False

        for turn in range(1, MAX_TURNS + 1):
            start = time.perf_counter()
            response = agent.respond(session_id, user_message, turn, TOP_K)
            turn_durations.append(time.perf_counter() - start)

            if turn >= RESCUE_AT:
                reached_rescue = True

            attribute = response.get("ask_attribute")
            attribute_counts[scenario_type][turn][attribute] += 1

            ranked = [r["parent_asin"] for r in response.get("recommendations", [])]
            recs_count_by_turn[turn].append(len(ranked))

            if override_applied and target in ranked:
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, ignore my earlier preference."))
            else:
                reply, boundary_used = customer_reply(effective_sample, attribute, disclosed, boundary_used)
                if session_first_no_pref_turn is None and "additional preference" in reply:
                    session_first_no_pref_turn = turn
                user_message = reply

        if session_first_no_pref_turn is not None:
            first_no_pref_turn_by_scenario[scenario_type].append(session_first_no_pref_turn)
        else:
            first_no_pref_none_count_by_scenario[scenario_type] += 1

        rescue_outcomes.append((scenario_type, reached_rescue, hit_turn))

    # ================= TABLES =================

    rows = [
        [
            str(turn),
            str(len(recs_count_by_turn[turn])),
            f"{statistics.fmean(recs_count_by_turn[turn]):.2f}",
            str(min(recs_count_by_turn[turn])),
            str(max(recs_count_by_turn[turn])),
        ]
        for turn in sorted(recs_count_by_turn)
    ]
    print_table(
        ["Turn", "Sessions reaching it", "Avg #recs", "Min", "Max"],
        rows,
        title="1. Recommendation count by turn (confidence gate behavior; contract allows <=10, not always ==10)",
    )

    all_attributes = sorted(
        {attr for turns in attribute_counts.values() for counts in turns.values() for attr in counts},
        key=lambda a: -sum(
            counts.get(a, 0) for turns in attribute_counts.values() for counts in turns.values()
        ),
    )
    for scenario_type in sorted(attribute_counts):
        turns = attribute_counts[scenario_type]
        rows = [
            [str(turn), *[str(turns[turn].get(attr, 0)) for attr in all_attributes]]
            for turn in sorted(turns)
        ]
        print_table(
            ["Turn", *all_attributes],
            rows,
            title=f"2. ask_attribute distribution -- scenario_type={scenario_type!r}",
        )

    rows = []
    for scenario_type in sorted(first_no_pref_turn_by_scenario):
        vals = first_no_pref_turn_by_scenario[scenario_type]
        rows.append([
            scenario_type,
            f"{statistics.median(vals):.1f}" if vals else "n/a",
            str(first_no_pref_none_count_by_scenario.get(scenario_type, 0)),
        ])
    print_table(
        ["Scenario", "Median 1st no-pref turn", "Never (hit or ran out first)"],
        rows,
        title="3. Turn the constraint pool stopped yielding new info, by scenario (n=200)",
    )

    rescue_by_scenario: dict[str, dict[str, int]] = defaultdict(lambda: {"reached": 0, "total": 0, "hit_in_rescue": 0})
    for scenario_type, reached, hit_turn in rescue_outcomes:
        bucket = rescue_by_scenario[scenario_type]
        bucket["total"] += 1
        if reached:
            bucket["reached"] += 1
            if hit_turn is not None and hit_turn >= RESCUE_AT:
                bucket["hit_in_rescue"] += 1
    rows = [
        [
            scenario_type,
            str(bucket["total"]),
            str(bucket["reached"]),
            f"{100 * bucket['reached'] / bucket['total']:.1f}%",
            str(bucket["hit_in_rescue"]),
        ]
        for scenario_type, bucket in sorted(rescue_by_scenario.items())
    ]
    print_table(
        ["Scenario", "Sessions", "Reached rescue mode (turn>=5)", "% reached", "Hit while in rescue mode"],
        rows,
        title="4. Rescue-mode activation: does the base agent usually resolve it before turn 5?",
    )

    print_table(
        ["Metric", "Seconds"],
        [
            ["count", str(len(turn_durations))],
            ["median", f"{statistics.median(turn_durations):.5f}"],
            ["p95", f"{percentile(turn_durations, 95):.5f}"],
            ["max", f"{max(turn_durations):.5f}"],
        ],
        title="5. Wall-clock time per turn (one Agent.respond() call)",
    )

    # ================= VERDICTS =================

    verdicts: list[tuple[str, str, str]] = []

    # -- Check 1: confidence gate sanity --
    # By design (see _gate_count): turn<=2 caps at <=2 recs, turn 3-4 is
    # variable, turn>=5 should always be exactly 10 (gate returns 10
    # unconditionally past turn 4). A turn>=5 session with <10 recs would
    # mean the candidate pool itself ran dry, not the gate -- worth knowing
    # either way.
    short_late_turns = [
        (turn, count) for turn in recs_count_by_turn for count in recs_count_by_turn[turn]
        if turn >= 5 and count < TOP_K
    ]
    if not short_late_turns:
        verdicts.append(("Check 1: confidence gate", "PASS", "turn>=5 always returns exactly 10 recs"))
    else:
        frac = len(short_late_turns) / sum(1 for t in recs_count_by_turn for _ in recs_count_by_turn[t] if t >= 5)
        verdict = "WARN" if frac < 0.05 else "FAIL"
        verdicts.append((
            "Check 1: confidence gate", verdict,
            f"{len(short_late_turns)} turn(s) at turn>=5 returned <10 recs ({frac:.1%}) -- candidate pool exhaustion, not the gate",
        ))

    # -- Check 2: constraint pool drain speed (same thresholds as before) --
    all_first_no_pref = [t for vals in first_no_pref_turn_by_scenario.values() for t in vals]
    overall_median_first_drain = statistics.median(all_first_no_pref) if all_first_no_pref else None
    if overall_median_first_drain is None:
        verdicts.append(("Check 2: drain speed", "WARN", "no session ever produced a 'no additional preference' reply"))
    elif overall_median_first_drain <= 4:
        verdicts.append(("Check 2: drain speed", "PASS", f"median first-drain turn = {overall_median_first_drain:.1f}"))
    elif overall_median_first_drain <= 5:
        verdicts.append(("Check 2: drain speed", "WARN", f"median first-drain turn = {overall_median_first_drain:.1f}"))
    else:
        verdicts.append(("Check 2: drain speed", "FAIL", f"median first-drain turn = {overall_median_first_drain:.1f}"))

    # -- Check 3: is rescue mode pulling its weight? --
    total_reached = sum(b["reached"] for b in rescue_by_scenario.values())
    total_hit_in_rescue = sum(b["hit_in_rescue"] for b in rescue_by_scenario.values())
    if total_reached == 0:
        verdicts.append(("Check 3: rescue mode value", "WARN", "no session ever reached rescue mode -- can't assess it"))
    else:
        rescue_hit_rate = total_hit_in_rescue / total_reached
        detail = f"{total_reached} session(s) reached rescue mode, {rescue_hit_rate:.1%} of those hit while in it"
        if rescue_hit_rate >= 0.5:
            verdicts.append(("Check 3: rescue mode value", "PASS", detail))
        elif rescue_hit_rate >= 0.2:
            verdicts.append(("Check 3: rescue mode value", "WARN", detail))
        else:
            verdicts.append(("Check 3: rescue mode value", "FAIL", detail))

    # -- Check 4: per-turn latency --
    median_ms = statistics.median(turn_durations) * 1000
    p95_ms = percentile(turn_durations, 95) * 1000
    detail = f"median={median_ms:.3f}ms  p95={p95_ms:.3f}ms"
    if median_ms < 50:
        verdicts.append(("Check 4: per-turn latency", "PASS", detail))
    elif median_ms < 150:
        verdicts.append(("Check 4: per-turn latency", "WARN", detail))
    else:
        verdicts.append(("Check 4: per-turn latency", "FAIL", detail))

    print("\n" + "=" * 60)
    print("VERDICTS")
    print("=" * 60)
    name_width = max(len(name) for name, _, _ in verdicts)
    for name, verdict, detail in verdicts:
        print(f"[{verdict:<4}] {name:<{name_width}}  {detail}")


if __name__ == "__main__":
    main()
