"""Runs the full 200-session public set through starter.fake_agent.Agent
(real DialogController from src.dialog, ranked via starter.stub_ranker),
prints five diagnostic tables, and then a VERDICTS summary block of four
PASS/WARN/FAIL checks against explicit thresholds.

Note: src.dialog.PENALTY_CAP is 0.99, not 0.6 -- 0.6 was a threshold used
by tests/test_invariants_full_run.py, not a cap enforced anywhere in the
code. Table 1 and Check 1 report the actual values reached so you can see
where they land relative to either number.

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
from starter.fake_agent import Agent, extract_constraints

import src.dialog as dialog_module

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10


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


# Wrap src.dialog's stub_rank reference so take_turn()'s internal ranking
# time can be subtracted out of "dialog module only" latency (Check 4).
# Single-threaded diagnostic script, so a module-level holder is safe.
_original_stub_rank = dialog_module.stub_rank
_last_rank_seconds = {"value": 0.0}


def _timed_stub_rank(*args, **kwargs):
    start = time.perf_counter()
    result = _original_stub_rank(*args, **kwargs)
    _last_rank_seconds["value"] = time.perf_counter() - start
    return result


dialog_module.stub_rank = _timed_stub_rank


def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    agent = Agent(CATALOG_PATH)

    # -- table/verdict data collectors --
    max_penalty_value_at_turn: dict[int, list[float]] = defaultdict(list)
    attribute_counts: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    exhaustion_turns: list[int | None] = []  # 'other' specifically exhausted
    first_no_pref_turn_by_scenario: dict[str, list[int]] = defaultdict(list)
    first_no_pref_none_count_by_scenario: Counter = Counter()
    other_exhausted_turn_by_scenario: dict[str, list[int]] = defaultdict(list)
    other_exhausted_none_count_by_scenario: Counter = Counter()
    override_constraint_counts: list[int] = []
    turn_durations: list[float] = []  # overall Agent.respond()-equivalent wall clock

    all_nonzero_penalties: list[float] = []
    frac_penalized_per_turn: list[float] = []
    turns_with_meaningful_penalty = 0
    total_turns_seen = 0
    max_feedback_penalty_seen = 0.0

    dialog_only_ms: list[float] = []
    observe_ms: list[float] = []
    state_ms: list[float] = []
    take_turn_minus_rank_ms: list[float] = []

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

        running_max = 0.0
        max_by_turn: dict[int, float] = {}
        session_other_exhausted_turn: int | None = None
        session_first_no_pref_turn: int | None = None
        constraints_at_override: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            wall_start = time.perf_counter()

            constraints = extract_constraints(user_message)

            t0 = time.perf_counter()
            agent.controller.observe(session_id, user_message, constraints)
            t1 = time.perf_counter()
            state = agent.controller.state(session_id)
            t2 = time.perf_counter()
            candidate_ids = [pid for pid in agent._catalog_ids if pid not in state["exclude_ids"]]
            t3 = time.perf_counter()
            turn_result = agent.controller.take_turn(session_id, candidate_ids, top_k=TOP_K)
            t4 = time.perf_counter()

            turn_durations.append(t4 - wall_start)
            observe_ms.append((t1 - t0) * 1000)
            state_ms.append((t2 - t1) * 1000)
            take_turn_only = (t4 - t3) - _last_rank_seconds["value"]
            take_turn_minus_rank_ms.append(max(take_turn_only, 0.0) * 1000)
            dialog_only_ms.append(
                (t1 - t0) * 1000 + (t2 - t1) * 1000 + max(take_turn_only, 0.0) * 1000
            )

            attribute = turn_result["ask_attribute"]
            attribute_counts[scenario_type][turn][attribute] += 1

            merged_penalties = state["penalties"]
            feedback_only = agent.controller.sessions[session_id]["feedback_penalties"]
            if feedback_only:
                max_feedback_penalty_seen = max(max_feedback_penalty_seen, max(feedback_only.values()))

            nonzero = [v for v in merged_penalties.values() if v > 0]
            all_nonzero_penalties.extend(nonzero)
            total_turns_seen += 1
            if any(v > 0.05 for v in merged_penalties.values()):
                turns_with_meaningful_penalty += 1
            if candidate_ids:
                frac_penalized_per_turn.append(len(merged_penalties) / len(candidate_ids))

            current_max = max(merged_penalties.values()) if merged_penalties else 0.0
            running_max = max(running_max, current_max)
            max_by_turn[turn] = running_max

            ranked = [r["parent_asin"] for r in turn_result["recommendations"]]
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                constraints_at_override = len(state["constraints"])
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, ignore my earlier preference."))
            else:
                reply, boundary_used = customer_reply(effective_sample, attribute, disclosed, boundary_used)
                if session_first_no_pref_turn is None and "additional preference" in reply:
                    session_first_no_pref_turn = turn
                if (
                    session_other_exhausted_turn is None
                    and attribute == "other"
                    and reply == "I don't have an additional preference for other."
                ):
                    session_other_exhausted_turn = turn
                user_message = reply

        if max_by_turn:
            overall_max = max(max_by_turn.values())
            reached_turn = min(t for t, v in max_by_turn.items() if v == overall_max)
            max_penalty_value_at_turn[reached_turn].append(overall_max)

        exhaustion_turns.append(session_other_exhausted_turn)
        if session_first_no_pref_turn is not None:
            first_no_pref_turn_by_scenario[scenario_type].append(session_first_no_pref_turn)
        else:
            first_no_pref_none_count_by_scenario[scenario_type] += 1
        if session_other_exhausted_turn is not None:
            other_exhausted_turn_by_scenario[scenario_type].append(session_other_exhausted_turn)
        else:
            other_exhausted_none_count_by_scenario[scenario_type] += 1

        if scenario_type == "intent_override" and constraints_at_override is not None:
            override_constraint_counts.append(constraints_at_override)

    # ================= TABLES =================

    rows = [
        [
            str(turn),
            str(len(max_penalty_value_at_turn[turn])),
            f"{statistics.fmean(max_penalty_value_at_turn[turn]):.3f}",
            f"{max(max_penalty_value_at_turn[turn]):.3f}",
        ]
        for turn in sorted(max_penalty_value_at_turn)
    ]
    print_table(
        ["Turn", "Sessions", "Avg max penalty", "Max max penalty"],
        rows,
        title="1. Turn at which each session's overall max penalty was first reached (n=200)",
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

    exhaustion_counter = Counter(exhaustion_turns)
    rows = [
        [str(turn), str(exhaustion_counter[turn])]
        for turn in sorted(t for t in exhaustion_counter if t is not None)
    ]
    rows.append(["never (hit target, or ran out of turns, first)", str(exhaustion_counter.get(None, 0))])
    print_table(
        ["Turn pool exhausted ('other')", "Sessions"],
        rows,
        title="3. Turn the constraint pool stopped yielding new info (n=200)",
    )

    if override_constraint_counts:
        counter = Counter(override_constraint_counts)
        rows = [[str(n), str(counter[n])] for n in sorted(counter)]
        print_table(
            ["Constraints gathered before override fired", "Sessions"],
            rows,
            title=f"4. intent_override: constraints accumulated pre-override (n={len(override_constraint_counts)})",
        )
        print(
            f"   median={statistics.median(override_constraint_counts)}  "
            f"mean={statistics.fmean(override_constraint_counts):.2f}"
        )
    else:
        print("\n4. No intent_override sessions reached their override turn.")

    print_table(
        ["Metric", "Seconds"],
        [
            ["count", str(len(turn_durations))],
            ["median", f"{statistics.median(turn_durations):.5f}"],
            ["p95", f"{percentile(turn_durations, 95):.5f}"],
            ["max", f"{max(turn_durations):.5f}"],
        ],
        title="5. Wall-clock time per turn (one full turn, incl. constraint extraction + ranking)",
    )

    # ================= VERDICTS =================

    verdicts: list[tuple[str, str, str]] = []  # (name, verdict, detail)

    # -- Check 1: penalty spread --
    frac_meaningful = turns_with_meaningful_penalty / total_turns_seen if total_turns_seen else 0.0
    median_frac_penalized = statistics.median(frac_penalized_per_turn) if frac_penalized_per_turn else 0.0
    print_table(
        ["Metric", "Value"],
        [
            ["non-zero penalty count", str(len(all_nonzero_penalties))],
            ["min", f"{min(all_nonzero_penalties):.4f}" if all_nonzero_penalties else "n/a"],
            ["median", f"{statistics.median(all_nonzero_penalties):.4f}" if all_nonzero_penalties else "n/a"],
            ["p95", f"{percentile(all_nonzero_penalties, 95):.4f}" if all_nonzero_penalties else "n/a"],
            ["max", f"{max(all_nonzero_penalties):.4f}" if all_nonzero_penalties else "n/a"],
            ["fraction of turns w/ any penalty > 0.05", f"{frac_meaningful:.3f}"],
            ["median fraction of candidates penalized/turn", f"{median_frac_penalized:.4f}"],
            ["max feedback_penalties value seen anywhere", f"{max_feedback_penalty_seen:.4f}"],
        ],
        title="Check 1 data -- penalty spread",
    )
    check1_fail_reasons = []
    if frac_meaningful < 0.20:
        check1_fail_reasons.append(f"only {frac_meaningful:.1%} of turns produced a meaningful penalty (< 20%)")
    if median_frac_penalized > 0.80:
        check1_fail_reasons.append(
            f"a typical turn penalizes {median_frac_penalized:.1%} of candidates (> 80%, indiscriminate)"
        )
    feedback_bound_ok = max_feedback_penalty_seen <= dialog_module.NEGATIVE_FEEDBACK_WEIGHT + 1e-9
    if not feedback_bound_ok:
        check1_fail_reasons.append(
            f"max feedback_penalties value {max_feedback_penalty_seen:.4f} exceeds "
            f"NEGATIVE_FEEDBACK_WEIGHT ({dialog_module.NEGATIVE_FEEDBACK_WEIGHT}) -- feedback "
            f"stacks across turns rather than being freshly recomputed each time"
        )
    if check1_fail_reasons:
        verdicts.append(("Check 1: penalty spread", "FAIL", "; ".join(check1_fail_reasons)))
    else:
        verdicts.append(("Check 1: penalty spread", "PASS", f"{frac_meaningful:.1%} of turns meaningful"))

    # -- Check 2: constraint pool drain speed --
    rows = []
    for scenario_type in sorted(
        set(first_no_pref_turn_by_scenario) | set(other_exhausted_turn_by_scenario)
    ):
        first_vals = first_no_pref_turn_by_scenario.get(scenario_type, [])
        other_vals = other_exhausted_turn_by_scenario.get(scenario_type, [])
        rows.append([
            scenario_type,
            f"{statistics.median(first_vals):.1f}" if first_vals else "n/a",
            str(first_no_pref_none_count_by_scenario.get(scenario_type, 0)),
            f"{statistics.median(other_vals):.1f}" if other_vals else "n/a",
            str(other_exhausted_none_count_by_scenario.get(scenario_type, 0)),
        ])
    print_table(
        ["Scenario", "Median 1st no-pref turn", "Never (1st)", "Median 'other' exhausted turn", "Never (other)"],
        rows,
        title="Check 2 data -- constraint pool drain speed, by scenario",
    )
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

    # -- Check 3: override preparation --
    if override_constraint_counts:
        median_gathered = statistics.median(override_constraint_counts)
        if median_gathered >= 2:
            verdicts.append(("Check 3: override prep", "PASS", f"median constraints gathered pre-override = {median_gathered}"))
        else:
            verdicts.append(("Check 3: override prep", "FAIL", f"median constraints gathered pre-override = {median_gathered} (turns 1-2 are unscored and should be spent gathering)"))
    else:
        verdicts.append(("Check 3: override prep", "WARN", "no intent_override session reached its override turn"))

    # -- Check 4: per-turn latency, dialog module only --
    median_dialog_ms = statistics.median(dialog_only_ms) if dialog_only_ms else 0.0
    p95_dialog_ms = percentile(dialog_only_ms, 95) if dialog_only_ms else 0.0
    detail = f"median={median_dialog_ms:.3f}ms  p95={p95_dialog_ms:.3f}ms"
    if median_dialog_ms < 50:
        verdicts.append(("Check 4: dialog latency", "PASS", detail))
    elif median_dialog_ms < 150:
        verdicts.append(("Check 4: dialog latency", "WARN", detail))
    else:
        breakdown = (
            f"observe median={statistics.median(observe_ms):.3f}ms, "
            f"state median={statistics.median(state_ms):.3f}ms, "
            f"take_turn(-rank) median={statistics.median(take_turn_minus_rank_ms):.3f}ms"
        )
        verdicts.append(("Check 4: dialog latency", "FAIL", f"{detail}; slowest breakdown: {breakdown}"))

    # ================= SUMMARY BLOCK =================
    print("\n" + "=" * 60)
    print("VERDICTS")
    print("=" * 60)
    name_width = max(len(name) for name, _, _ in verdicts)
    for name, verdict, detail in verdicts:
        print(f"[{verdict:<4}] {name:<{name_width}}  {detail}")


if __name__ == "__main__":
    main()
