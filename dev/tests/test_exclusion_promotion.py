from __future__ import annotations

import unittest
from itertools import combinations

from src.dialog import DialogController
from legacy_starter.stub_ranker import rank

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "x",
}

CANDIDATE_IDS = [f"P{i:02d}" for i in range(1, 21)]

# Non-intent_override messages: turn 1 matches the 'exploring' template,
# later turns are ordinary customer replies. None of them match
# detect_scenario()'s intent_override shape, so every slate should be
# hard-excluded on the next observe().
MESSAGES = [
    "I'm looking for Jewelry Earrings, but I'm still exploring.",
    "For that, what matters is: silver; hoop.",
    "For that, what matters is: lightweight.",
]


class ExclusionPromotionTest(unittest.TestCase):
    def test_three_turns_never_repeat_a_product(self) -> None:
        controller = DialogController()
        controller.reset("session-1", PROFILE)

        slates: list[list[str]] = []
        for message in MESSAGES:
            controller.observe("session-1", message, constraints=[])
            state = controller.state("session-1")
            slate = rank(CANDIDATE_IDS, state, k=5)
            controller.register_slate("session-1", slate)
            slates.append(slate)

        for slate in slates:
            self.assertEqual(len(slate), 5)

        for (i, slate_a), (j, slate_b) in combinations(enumerate(slates), 2):
            overlap = set(slate_a) & set(slate_b)
            self.assertEqual(overlap, set(), f"slate {i} and slate {j} share {overlap}")

        # sanity: promotion actually landed in exclude_ids, not penalties.
        final_state = controller.state("session-1")
        self.assertTrue(set(slates[0]) <= final_state["exclude_ids"])
        self.assertTrue(set(slates[1]) <= final_state["exclude_ids"])
        self.assertEqual(final_state["penalties"], {})


if __name__ == "__main__":
    unittest.main()
