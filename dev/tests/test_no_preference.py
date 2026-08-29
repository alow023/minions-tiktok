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


class NoPreferenceTest(unittest.TestCase):
    def test_boundary_reply_sets_flag_but_not_exhausted(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)

        # literal template from evaluator.local_evaluator.customer_reply
        message = "I don't have a preference for material; please use your judgment."
        controller.observe("s1", message, [])

        state = controller.sessions["s1"]
        self.assertTrue(state["boundary_seen"])
        self.assertEqual(state["exhausted"], set())

    def test_exhausted_reply_marks_attribute_exhausted_and_not_boundary(self) -> None:
        controller = DialogController()
        controller.reset("s2", PROFILE)

        # literal template from evaluator.local_evaluator.customer_reply
        message = "I don't have an additional preference for material."
        controller.observe("s2", message, [])

        state = controller.sessions["s2"]
        self.assertIn("material", state["exhausted"])
        self.assertFalse(state["boundary_seen"])

    def test_choose_question_skips_exhausted_attributes(self) -> None:
        # 'feature' is the only one of the seven askable attributes left
        # un-exhausted below, so it must win regardless of its Balance
        # score -- give it a clean 50/50 split so it also clears the score
        # threshold on its own merits.
        candidate_attributes = {
            f"C{i}": {"feature": "a" if i % 2 == 0 else "b"} for i in range(10)
        }
        controller = DialogController(candidate_attributes)
        controller.reset("s3", PROFILE)
        controller.observe("s3", "I don't have an additional preference for other.", [])
        for attribute in ("budget", "material", "color", "size", "style", "use_case"):
            controller.observe("s3", f"I don't have an additional preference for {attribute}.", [])

        result = controller.choose_question("s3", candidate_ids=list(candidate_attributes.keys()))
        self.assertEqual(result, "feature")


if __name__ == "__main__":
    unittest.main()
