from __future__ import annotations

import logging
import unittest

from src.dialog import BALANCE_THRESHOLD, EMPIRICAL_ATTRIBUTE_PRIORS, DialogController

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "x",
}


def _capture_logs(logger_name: str, fn) -> list[logging.LogRecord]:
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
    return captured


class EmpiricalPriorsTest(unittest.TestCase):
    def test_priors_sum_to_roughly_one(self) -> None:
        self.assertAlmostEqual(sum(EMPIRICAL_ATTRIBUTE_PRIORS.values()), 1.0, places=2)

    def test_feature_has_highest_prior_and_clears_threshold_even_at_neutral_balance(self) -> None:
        # 'feature' is the catch-all default in classify_constraint, so
        # empirically it dominates -- see diagnose_buckets.py's output.
        # Its prior is high enough that even a neutral (estimated) Balance
        # of 0.5 -- i.e. zero real data -- still clears BALANCE_THRESHOLD:
        # 0.5 * 0.5256 = 0.2628 >= 0.2.
        self.assertEqual(max(EMPIRICAL_ATTRIBUTE_PRIORS, key=EMPIRICAL_ATTRIBUTE_PRIORS.get), "feature")
        self.assertGreaterEqual(0.5 * EMPIRICAL_ATTRIBUTE_PRIORS["feature"], BALANCE_THRESHOLD)

    def test_no_data_bucket_uses_estimated_neutral_balance_not_prior_directly(self) -> None:
        # No candidate has any attribute data at all: every bucket's
        # Balance is estimated at a neutral 0.5, so score = 0.5 * prior for
        # each. 'feature' still wins (0.2628, the highest such score) and
        # clears BALANCE_THRESHOLD.
        candidate_ids = [f"C{i}" for i in range(20)]
        controller = DialogController({})
        controller.reset("s1", PROFILE)

        result = controller.choose_question("s1", candidate_ids)
        self.assertEqual(result, "feature")

    def test_empty_coverage_is_accumulated_silently_not_logged_per_call(self) -> None:
        candidate_ids = [f"C{i}" for i in range(20)]
        controller = DialogController({})

        def run_many_turns() -> None:
            for i in range(5):
                controller.reset(f"s{i}", PROFILE)
                controller.choose_question(f"s{i}", candidate_ids)

        # Five sessions all hit the same zero-coverage buckets repeatedly.
        # None of that should produce a log line by itself -- only counted.
        logs_during_calls = _capture_logs("src.dialog", run_many_turns)
        self.assertEqual(logs_during_calls, [])

        # The counts were tracked even though nothing was logged.
        self.assertEqual(controller._empty_coverage_counts["feature"], 5)
        self.assertEqual(controller._empty_coverage_counts["budget"], 5)

    def test_log_empty_bucket_summary_emits_one_aggregated_warning(self) -> None:
        candidate_ids = [f"C{i}" for i in range(20)]
        controller = DialogController({})
        controller.reset("s1", PROFILE)
        controller.choose_question("s1", candidate_ids)

        with self.assertLogs("src.dialog", level="WARNING") as logs:
            controller.log_empty_bucket_summary()

        self.assertEqual(len(logs.output), 1)
        message = logs.output[0]
        # every attribute that had zero coverage should be named once, in
        # a single aggregated line rather than one warning apiece.
        for attribute in ("budget", "material", "color", "size", "style", "use_case", "feature"):
            self.assertIn(attribute, message)


if __name__ == "__main__":
    unittest.main()
