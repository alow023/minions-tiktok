from __future__ import annotations

import unittest

from src.dialog import DialogController

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "x",
}


class TurnPolicyTest(unittest.TestCase):
    def test_tops_up_to_exactly_top_k_when_candidate_pool_is_thin(self) -> None:
        # Only 3 candidates offered this turn, but the controller knows
        # about 50 products total -- take_turn should still return 10.
        candidate_attributes = {f"P{i}": {} for i in range(50)}
        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)
        controller.observe("s1", "I'm looking for Shirts, but I'm still exploring.", [])

        turn = controller.take_turn("s1", candidate_ids=["P0", "P1", "P2"], top_k=10)

        self.assertEqual(len(turn["recommendations"]), 10)
        self.assertIsNotNone(turn["ask_attribute"])
        shown = {rec["parent_asin"] for rec in turn["recommendations"]}
        self.assertIn("P0", shown)
        self.assertIn("P1", shown)
        self.assertIn("P2", shown)

    def test_top_up_never_reintroduces_an_excluded_product(self) -> None:
        candidate_attributes = {f"P{i}": {} for i in range(15)}
        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)
        controller.observe("s1", "I'm looking for Shirts, but I'm still exploring.", [])
        controller.register_slate("s1", [f"P{i}" for i in range(10)])
        # promotes P0..P9 into exclude_ids (not an intent_override session).
        controller.observe("s1", "For that, what matters is: something.", [])

        turn = controller.take_turn("s1", candidate_ids=["P10"], top_k=10)

        shown = {rec["parent_asin"] for rec in turn["recommendations"]}
        self.assertEqual(len(turn["recommendations"]), 5)  # only 5 unexcluded products exist
        self.assertTrue(shown.isdisjoint({f"P{i}" for i in range(10)}))

    def test_prefers_other_on_first_two_turns_of_intent_override(self) -> None:
        candidate_attributes = {f"P{i}": {"color": "blue" if i % 2 else "red"} for i in range(20)}
        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)
        candidate_ids = list(candidate_attributes.keys())

        # turn 1: intent_override initial message template.
        controller.observe("s1", "I'm looking for Accessories Belts. Buckle closure", [])
        turn1 = controller.take_turn("s1", candidate_ids)
        self.assertEqual(turn1["ask_attribute"], "other")

        # turn 2: still before the override (fires at turn 3 or 4).
        controller.observe("s1", "For that, what matters is: something.", [])
        turn2 = controller.take_turn("s1", candidate_ids)
        self.assertEqual(turn2["ask_attribute"], "other")

        # turn 3: no override message arrived, but the "first two turns"
        # carve-out has passed -- choose_question drives it now.
        controller.observe("s1", "For that, what matters is: something else.", [])
        turn3 = controller.take_turn("s1", candidate_ids)
        self.assertIsNotNone(turn3["ask_attribute"])


if __name__ == "__main__":
    unittest.main()
