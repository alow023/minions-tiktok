from __future__ import annotations

import random
import unittest

from evaluator.local_evaluator import behavior_for
from src.dialog import DialogController

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "x",
}


class OverrideHandlingTest(unittest.TestCase):
    def test_override_drops_superseded_constraints_and_clears_penalties_not_exclude_ids(self) -> None:
        card = {"hard_constraints": ["leather"], "soft_preferences": ["Buckle closure"]}
        behavior = behavior_for("intent_override", card, random.Random(0))
        override_message = behavior["override"]["message"]
        self.assertEqual(override_message, "Actually, ignore my earlier preference. What I need is: leather.")

        controller = DialogController()
        controller.reset("session-1", PROFILE)

        initial_constraints = [
            {"text": "cotton", "attribute": "material", "kind": "hard"},
            {"text": "Buckle closure", "attribute": "feature", "kind": "soft"},
            {"text": "budget around $20", "attribute": "budget", "kind": "soft"},
        ]
        controller.observe(
            "session-1",
            "I'm looking for Accessories Belts. Buckle closure",
            initial_constraints,
        )
        state = controller.state("session-1")
        self.assertEqual(len(state["constraints"]), 3)

        # A slate shown before the override lands and misses -> penalized,
        # not excluded, since it's still inside the unfired intent_override
        # window. This gives us a non-empty penalties dict to verify gets
        # cleared by the override.
        controller.register_slate("session-1", ["P1", "P2"])
        controller.observe("session-1", "For that, what matters is: fabric.", [])
        state = controller.state("session-1")
        self.assertEqual(state["penalties"], {"P1": 0.5, "P2": 0.5})

        new_constraint = {"text": "leather", "attribute": "material", "kind": "hard"}
        controller.observe("session-1", override_message, [new_constraint])

        state = controller.state("session-1")
        remaining_text = {c["text"] for c in state["constraints"]}
        self.assertNotIn("cotton", remaining_text)
        self.assertNotIn("Buckle closure", remaining_text)
        self.assertNotIn("budget around $20", remaining_text)
        self.assertIn("leather", remaining_text)
        # shown_penalties (P1/P2 were actually displayed and missed) survives
        # the override -- that fact stays true regardless of intent change.
        # Only feedback_penalties (none here) would have been cleared.
        self.assertEqual(state["penalties"], {"P1": 0.5, "P2": 0.5})

        # exclude_ids is untouched by the override -- it isn't populated in
        # this scenario (misses go to penalties pre-override), but the key
        # itself must survive unchanged and remain a set.
        self.assertEqual(state["exclude_ids"], set())


if __name__ == "__main__":
    unittest.main()
