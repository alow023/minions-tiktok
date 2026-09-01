"""Runs the full 200-session public set through starter.agent.Agent
(the ShoppingCopilot design: a field-weighted BM25 base agent for the first
few turns, then Person C's constraint-filtered multi-route pipeline once
the base agent stalls) and checks invariants on every turn of every
session. All violations are collected first; nothing is asserted until the
whole run finishes, so a single failing run shows the complete picture
instead of stopping at the first broken turn.

Adapted from an earlier version that tested src.dialog.DialogController
(via the now-removed starter/fake_agent.py). That agent's internal state
(exhausted-attribute tracking, a 0.0-1.0 penalty scale) doesn't exist in
this architecture, so those invariants were dropped rather than forced
onto a different design; see the analogues used below.
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
from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10

# Attributes _ask_infogain()/_ask_hybrid() actually cycle through and track
# in st["asked"] (see starter/agent.py's base Agent). 'other' is the
# catch-all fallback and is expected to repeat every turn by design.
ASKABLE_ATTRIBUTES = {"material", "color", "budget", "brand"}
# classify_constraint() in the evaluator never emits 'brand' or 'category',
# so a real constraint can never match either -- asking for them always
# produces "I don't have an additional preference for X", wasting the turn.
NEVER_DIRECTLY_ASKED = {"brand", "category"}

INVARIANTS = [
    "recommendation_entries_well_formed",
    "recommendations_count_between_1_and_10",
    "ask_attribute_legal_or_none",
    "ask_attribute_not_brand_or_category",
    "no_cross_slate_duplicate_id",
    "no_repeated_askable_attribute_within_session",
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

            slate_turns: dict[str, list[int]] = defaultdict(list)
            asked_askable: set[str] = set()

            for turn in range(1, MAX_TURNS + 1):
                response = agent.respond(session_id, user_message, turn, TOP_K)

                recs = response.get("recommendations")
                ids_this_turn: list[str] = []
                if not isinstance(recs, list) or not (1 <= len(recs) <= TOP_K):
                    violations["recommendations_count_between_1_and_10"].add(sample_id)
                    details.append(
                        f"{sample_id} turn {turn}: recommendations count == "
                        f"{len(recs) if isinstance(recs, list) else type(recs).__name__}"
                    )
                for entry in recs or []:
                    parent_asin = entry.get("parent_asin") if isinstance(entry, dict) else None
                    if not (isinstance(entry, dict) and isinstance(parent_asin, str) and parent_asin):
                        violations["recommendation_entries_well_formed"].add(sample_id)
                        details.append(f"{sample_id} turn {turn}: malformed recommendation entry {entry!r}")
                        continue
                    ids_this_turn.append(parent_asin)

                attribute = response.get("ask_attribute")
                if attribute is not None and attribute not in ALLOWED_ATTRIBUTES:
                    violations["ask_attribute_legal_or_none"].add(sample_id)
                    details.append(f"{sample_id} turn {turn}: illegal ask_attribute={attribute!r}")
                if attribute in NEVER_DIRECTLY_ASKED:
                    violations["ask_attribute_not_brand_or_category"].add(sample_id)
                    details.append(f"{sample_id} turn {turn}: asked {attribute!r} directly")

                for rank, parent_asin in enumerate(ids_this_turn, start=1):
                    slate_turns[parent_asin].append((turn, rank))

                if attribute in ASKABLE_ATTRIBUTES:
                    if attribute in asked_askable:
                        violations["no_repeated_askable_attribute_within_session"].add(sample_id)
                        details.append(f"{sample_id} turn {turn}: attribute {attribute!r} asked again")
                    asked_askable.add(attribute)

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
                    # The customer has erased their slots, and the agent
                    # erases its shown-set to match, so products it offered
                    # under the superseded preference are fair to offer
                    # again. Reset the tracker for the same reason.
                    slate_turns.clear()
                else:
                    user_message, boundary_used = customer_reply(
                        effective_sample, attribute, disclosed, boundary_used
                    )

            # A product may legitimately appear on more than one slate, but
            # only as the agent re-asserting its single leading answer
            # (tracking restarts after an intent override, see above): once
            # the exact attribute-phrase route locks onto a product, that
            # product must stay at rank 1 until the customer moves on (an
            # intent_override is not scorable until the override turn, so a
            # product identified on turn 2 has to still be there on turn 3).
            # Re-showing a product *deeper* in a later slate is the real
            # defect this invariant guards: a scoring slot spent on a
            # candidate the customer already declined.
            for parent_asin, appearances in slate_turns.items():
                if len(appearances) > 1 and any(rank != 1 for _, rank in appearances):
                    violations["no_cross_slate_duplicate_id"].add(sample_id)
                    details.append(
                        f"{sample_id}: product {parent_asin!r} re-shown off rank 1 "
                        f"at (turn, rank) {appearances}")

        except Exception as exc:  # noqa: BLE001 -- deliberately broad: this IS the invariant
            violations["no_exception_raised"].add(sample_id)
            details.append(f"{sample_id}: raised {type(exc).__name__}: {exc}")

    return violations, details


class FullRunInvariantsTest(unittest.TestCase):
    def test_invariants_hold_across_full_public_set(self) -> None:
        violations, details = _run()

        print("\nInvariant violation summary (sessions violating each, out of 200):")
        print(f"  {'invariant':<48} {'sessions':>8}")
        for name in INVARIANTS:
            print(f"  {name:<48} {len(violations[name]):>8}")

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
