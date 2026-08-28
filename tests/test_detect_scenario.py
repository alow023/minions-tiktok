from __future__ import annotations

import unittest

from evaluator.local_evaluator import initial_message
from src.dialog import detect_scenario


class DetectScenarioTest(unittest.TestCase):
    def test_buying_template(self) -> None:
        sample = {"scenario_type": "buying", "intent_card": {"hard_constraints": ["cotton"]}}
        message = initial_message(sample, "Shirts T-Shirts", set())
        self.assertEqual(message, "I'm looking for Shirts T-Shirts. A key requirement is: cotton.")
        self.assertEqual(detect_scenario(message), "buying")

    def test_intent_override_template(self) -> None:
        sample = {"scenario_type": "intent_override", "behavior": {"override": {"old_value": "Buckle closure"}}}
        message = initial_message(sample, "Accessories Belts", set())
        self.assertEqual(message, "I'm looking for Accessories Belts. Buckle closure")
        self.assertEqual(detect_scenario(message), "intent_override")

    def test_browsing_and_boundary_share_the_exploring_template(self) -> None:
        for scenario_type in ("browsing", "boundary"):
            sample = {"scenario_type": scenario_type}
            message = initial_message(sample, "Jewelry Earrings", set())
            self.assertEqual(message, "I'm looking for Jewelry Earrings, but I'm still exploring.")
            self.assertEqual(detect_scenario(message), "exploring")


if __name__ == "__main__":
    unittest.main()
