from __future__ import annotations

import unittest

from src.dialog import LEGAL_ATTRIBUTES, DialogController


class NeverNoneTest(unittest.TestCase):
    def test_choose_question_always_returns_a_legal_attribute(self) -> None:
        controller = DialogController()
        controller.reset("session-with-history", {
            "purchase_frequency": "3-4 prior purchases",
            "average_prior_rating": 4.5,
            "rating_style": "usually positive",
            "preference_tags": ["fit", "comfort"],
            "summary": "x",
        })

        results = []
        for i in range(12):
            session_id = "session-with-history" if i % 2 == 0 else "never-reset-session"
            result = controller.choose_question(session_id, candidate_ids=[])
            results.append(result)

        self.assertEqual(len(results), 12)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn(result, LEGAL_ATTRIBUTES)


if __name__ == "__main__":
    unittest.main()
