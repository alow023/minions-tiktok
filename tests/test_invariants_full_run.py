"""Runs the full 200-session public set through starter.fake_agent.Agent
(real DialogController from src.dialog, ranked via starter.stub_ranker) and
checks nine invariants on every single turn of every session. All
violations are collected first; nothing is asserted until the whole run
finishes, so a single failing run shows the complete picture instead of
stopping at the first broken turn.
"""

from __future__ import annotations

import unittest
from collections import defaultdict

from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.fake_agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10
PENALTY_MAX = 0.6

# The 7 buckets classify_constraint() can actually emit / choose_question()
# tracks in 'asked' and 'exhausted'. 'other', 'brand', and 'category' are
# deliberately excluded from the "never repeat" bookkeeping -- 'other' is
# meant to repeat every turn by design, and 'brand'/'category' are covered
# by their own dedicated invariant instead.
ASKABLE_ATTRIBUTES = {"budget", "material", "color", "size", "style", "use_case", "feature"}
NEVER_DIRECTLY_ASKED = {"brand", "category"}

INVARIANTS = [
    "recommendations_count_is_10",
    "recommendation_entries_well_formed",
    "ask_attribute_non_null_and_legal",
    "ask_attribute_not_brand_or_category",
    "no_cross_slate_duplicate_id",
    "no_repeated_attribute_within_epoch",
    "no_exhausted_attribute_reasked",
    "no_penalty_over_0.6",
    "no_exception_raised",
]


def _run() -> tuple[dict[str, set[str]], list[str]]:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)

    violations: dict[str, set[str]] = {name: set() for name in INVARIANTS}
    details: list[str] = []

    agent = Agent(CATALOG_PATH)

    for sample in samples:
        sample_id = sample["sample_id"]
        session_id = f"inv_{sample_id}"
        scenario_type = sample["scenario_type"]

        try:
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

            # product_id -> list of (turn, pre_override_flag)
            slate_occurrences: dict[str, list[tuple[int, bool]]] = defaultdict(list)
            # (attribute, override_epoch) pairs already asked
            asked_in_epoch: set[tuple[str, int]] = set()

            for turn in range(1, MAX_TURNS + 1):
                response = agent.respond(session_id, user_message, turn, TOP_K)

                recs = response.get("recommendations")
                ids_this_turn: list[str] = []
                if not isinstance(recs, list) or len(recs) != TOP_K:
                    violations["recommendations_count_is_10"].add(sample_id)
                    details.append(
                        f"{sample_id} turn {turn}: recommendations count == "
                        f"{len(recs) if isinstance(recs, list) else type(recs).__name__}, expected {TOP_K}"
                    )
                for entry in recs or []:
                    parent_asin = entry.get("parent_asin") if isinstance(entry, dict) else None
                    if not (isinstance(entry, dict) and isinstance(parent_asin, str) and parent_asin):
                        violations["recommendation_entries_well_formed"].add(sample_id)
                        details.append(f"{sample_id} turn {turn}: malformed recommendation entry {entry!r}")
                        continue
                    ids_this_turn.append(parent_asin)

                attribute = response.get("ask_attribute")
                if attribute is None or attribute not in ALLOWED_ATTRIBUTES:
                    violations["ask_attribute_non_null_and_legal"].add(sample_id)
                    details.append(f"{sample_id} turn {turn}: ask_attribute={attribute!r}")
                if attribute in NEVER_DIRECTLY_ASKED:
                    violations["ask_attribute_not_brand_or_category"].add(sample_id)
                    details.append(f"{sample_id} turn {turn}: asked {attribute!r} directly")

                # Internal controller state, read for whitebox invariant
                # checks not exposed by DialogController.state()'s 4 keys.
                internal = agent.controller.sessions.get(session_id, {})
                override_fired_now = bool(internal.get("override_fired"))
                session_scenario = internal.get("scenario")
                exhausted_now = set(internal.get("exhausted") or set())
                pre_override = (session_scenario == "intent_override") and not override_fired_now
                epoch = int(override_fired_now)

                for parent_asin in ids_this_turn:
                    slate_occurrences[parent_asin].append((turn, pre_override))

                if attribute in ASKABLE_ATTRIBUTES:
                    key = (attribute, epoch)
                    if key in asked_in_epoch:
                        violations["no_repeated_attribute_within_epoch"].add(sample_id)
                        details.append(
                            f"{sample_id} turn {turn}: attribute {attribute!r} asked again "
                            f"within the same override epoch ({epoch})"
                        )
                    asked_in_epoch.add(key)

                if attribute in exhausted_now:
                    violations["no_exhausted_attribute_reasked"].add(sample_id)
                    details.append(f"{sample_id} turn {turn}: asked already-exhausted attribute {attribute!r}")

                penalties_now = agent.controller.state(session_id)["penalties"]
                over_limit = {pid: val for pid, val in penalties_now.items() if val > PENALTY_MAX}
                if over_limit:
                    violations["no_penalty_over_0.6"].add(sample_id)
                    worst = max(over_limit.values())
                    details.append(
                        f"{sample_id} turn {turn}: {len(over_limit)} product(s) with penalty > {PENALTY_MAX} "
                        f"(worst={worst:.4f})"
                    )

                ranked = ids_this_turn
                if override_applied and target in ranked:
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
                    user_message, boundary_used = customer_reply(
                        effective_sample, attribute, disclosed, boundary_used
                    )

            for parent_asin, occurrences in slate_occurrences.items():
                if len(occurrences) <= 1:
                    continue
                if any(not pre for _turn, pre in occurrences):
                    violations["no_cross_slate_duplicate_id"].add(sample_id)
                    turns = [t for t, _ in occurrences]
                    details.append(
                        f"{sample_id}: product {parent_asin!r} shown in turns {turns} "
                        f"(not all pre-override in an intent_override session)"
                    )

        except Exception as exc:  # noqa: BLE001 -- deliberately broad: this IS the invariant
            violations["no_exception_raised"].add(sample_id)
            details.append(f"{sample_id}: raised {type(exc).__name__}: {exc}")

    return violations, details


class FullRunInvariantsTest(unittest.TestCase):
    def test_invariants_hold_across_full_public_set(self) -> None:
        violations, details = _run()

        print("\nInvariant violation summary (sessions violating each, out of 200):")
        print(f"  {'invariant':<38} {'sessions':>8}")
        for name in INVARIANTS:
            print(f"  {name:<38} {len(violations[name]):>8}")

        if details:
            print(f"\n{len(details)} violation detail line(s) (showing up to 40):")
            for line in details[:40]:
                print(f"  - {line}")
            if len(details) > 40:
                print(f"  ... and {len(details) - 40} more")

        failing = {name: sessions for name, sessions in violations.items() if sessions}
        self.assertEqual(
            failing,
            {},
            f"{len(failing)} invariant(s) violated: " + ", ".join(f"{n}={len(s)}" for n, s in failing.items()),
        )


if __name__ == "__main__":
    unittest.main()
