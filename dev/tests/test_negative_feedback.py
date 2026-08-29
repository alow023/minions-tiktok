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


class NegativeFeedbackTest(unittest.TestCase):
    def test_over_represented_rejected_value_penalizes_matching_survivors_more(self) -> None:
        rejected_ids = [f"R{i}" for i in range(10)]  # all black, all rejected
        black_survivors = [f"B{i}" for i in range(10)]
        red_survivors = [f"Rd{i}" for i in range(10)]

        candidate_attributes: dict[str, dict[str, str]] = {}
        for pid in rejected_ids + black_survivors:
            candidate_attributes[pid] = {"color": "black"}
        for pid in red_survivors:
            candidate_attributes[pid] = {"color": "red"}

        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)

        # turn 1: no prior slate yet, just establishes scenario.
        controller.observe("s1", "I'm looking for Jewelry Earrings, but I'm still exploring.", [])
        controller.register_slate("s1", rejected_ids)

        # turn 2: the rejected slate is promoted (hard-excluded, since this
        # isn't an unfired intent_override session) and negative feedback
        # is computed against the survivors (black_survivors + red_survivors,
        # 20 total -- rejected_ids are no longer "surviving").
        controller.observe("s1", "For that, what matters is: something else.", [])
        state = controller.state("s1")

        # freq_rejected(black) = 10/10 = 1.0
        # freq_surviving(black) = 10/20 = 0.5
        # diff = 0.5 -> increment = 0.25 * 0.5 = 0.125
        for pid in black_survivors:
            self.assertAlmostEqual(state["penalties"][pid], 0.125)
        # red was not over-represented among rejects (0 rejects were red),
        # so red survivors get no penalty at all.
        for pid in red_survivors:
            self.assertEqual(state["penalties"].get(pid, 0.0), 0.0)

        for black_pid in black_survivors:
            for red_pid in red_survivors:
                self.assertGreater(state["penalties"][black_pid], state["penalties"].get(red_pid, 0.0))

        self.assertTrue(all(penalty < 1.0 for penalty in state["penalties"].values()))
        # rejected items themselves were hard-excluded, not merely penalized.
        self.assertEqual(state["exclude_ids"], set(rejected_ids))


if __name__ == "__main__":
    unittest.main()
