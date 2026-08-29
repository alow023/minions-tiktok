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


class PenaltySplitTest(unittest.TestCase):
    def test_shown_penalty_survives_override_feedback_penalty_does_not(self) -> None:
        # Enough candidate data to give _apply_negative_feedback a real
        # over-represented value to react to: 5 rejected items all black,
        # 5 surviving black, 5 surviving red.
        candidate_attributes = {}
        for i in range(5):
            candidate_attributes[f"R{i}"] = {"color": "black"}  # rejected slate
            candidate_attributes[f"B{i}"] = {"color": "black"}  # survivors
            candidate_attributes[f"Rd{i}"] = {"color": "red"}   # survivors

        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)

        # turn 1: intent_override initial message, no prior slate yet.
        controller.observe("s1", "I'm looking for Accessories Belts. Buckle closure", [])
        controller.register_slate("s1", [f"R{i}" for i in range(5)])

        # turn 2: still pre-override. Promoting the rejected slate here
        # produces a shown_penalty (flat +0.5, since misses don't count yet
        # in an unfired intent_override session) AND a feedback_penalty on
        # the black survivors (Rocchio negative feedback: black was
        # over-represented among the rejects).
        controller.observe("s1", "For that, what matters is: something.", [])
        state = controller.state("s1")
        internal = controller.sessions["s1"]

        # R0..R4 got the flat shown_penalty exactly (component-level check).
        # They ALSO remain "surviving" (pre-override misses aren't
        # excluded), so as black items among an over-represented black
        # rejected slate they pick up their own feedback_penalty too --
        # the merged view is therefore >= 0.5, not necessarily == 0.5.
        for i in range(5):
            self.assertAlmostEqual(internal["shown_penalties"][f"R{i}"], 0.5)
            self.assertGreaterEqual(state["penalties"][f"R{i}"], 0.5)
        # B0..B4 (black survivors, never shown) got a pure feedback_penalty
        # from Rocchio, no shown_penalty at all.
        for i in range(5):
            self.assertNotIn(f"B{i}", internal["shown_penalties"])
            self.assertGreater(state["penalties"].get(f"B{i}", 0.0), 0.0)
        # Rd0..Rd4 (red survivors) got nothing -- red wasn't over-represented.
        for i in range(5):
            self.assertEqual(state["penalties"].get(f"Rd{i}", 0.0), 0.0)

        shown_before = dict(controller.sessions["s1"]["shown_penalties"])
        feedback_before = dict(controller.sessions["s1"]["feedback_penalties"])
        self.assertTrue(shown_before)
        self.assertTrue(feedback_before)

        # turn 3: the override fires.
        controller.observe(
            "s1", "Actually, ignore my earlier preference. What I need is: cotton.", []
        )

        self.assertEqual(controller.sessions["s1"]["shown_penalties"], shown_before)
        self.assertEqual(controller.sessions["s1"]["feedback_penalties"], {})

        state = controller.state("s1")
        for parent_asin, penalty in shown_before.items():
            self.assertEqual(state["penalties"][parent_asin], penalty)
        for parent_asin in feedback_before:
            if parent_asin not in shown_before:
                self.assertEqual(state["penalties"].get(parent_asin, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
