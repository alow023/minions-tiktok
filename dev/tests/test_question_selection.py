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


def _build_candidates() -> dict[str, dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    for i in range(100):
        parent_asin = f"C{i:03d}"
        candidates[parent_asin] = {
            # color: perfect 50/50 split -> Balance = 1.0 (real, measured)
            "color": "blue" if i < 50 else "red",
            # material: 90/10 split -> Balance = 0.2 (real, measured)
            "material": "cotton" if i < 90 else "leather",
        }
        # nothing else ('feature', 'style', 'size', 'budget', 'use_case')
        # is populated at all -- their Balance has to be *estimated* at 0.5.
    return candidates


class QuestionSelectionTest(unittest.TestCase):
    def test_feature_outranks_a_perfect_color_split(self) -> None:
        # score(a) = Balance(a) * EMPIRICAL_ATTRIBUTE_PRIORS[a]:
        #   color:    Balance=1.0 (measured, perfect split) * prior 0.1263 = 0.1263
        #   feature:  Balance=0.5 (estimated, no data at all) * prior 0.5256 = 0.2628
        # Even though color splits these candidates perfectly and feature's
        # Balance is a mere neutral guess, feature's prior is more than 4x
        # color's -- customers essentially always have *something* to say
        # that lands in the 'feature' catch-all bucket (52.6% of real
        # constraints, per diagnose_buckets.py), while a color preference
        # is comparatively rare (12.6%) even when it happens to be
        # perfectly diagnostic for this particular candidate set. Expected
        # information gain, not raw split quality, is what's being ranked.
        candidate_attributes = _build_candidates()
        candidate_ids = list(candidate_attributes.keys())
        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)

        first = controller.choose_question("s1", candidate_ids)
        self.assertEqual(first, "feature")

        # Nothing else clears BALANCE_THRESHOLD (0.2) on this candidate
        # set: color caps out at 0.1263 and material at ~0.0547, both
        # bounded above by their own (small) priors regardless of how
        # balanced they are. So every subsequent call falls through to
        # 'other' rather than picking color or material outright.
        second = controller.choose_question("s1", candidate_ids)
        self.assertEqual(second, "other")

    def test_never_repeats_an_attribute_until_override_resets_asked(self) -> None:
        candidate_attributes = _build_candidates()
        candidate_ids = list(candidate_attributes.keys())
        controller = DialogController(candidate_attributes)
        controller.reset("s2", PROFILE)

        seen = {controller.choose_question("s2", candidate_ids) for _ in range(5)}
        # 'feature' wins once (see test above), then every remaining call
        # falls through to 'other' since nothing else clears the threshold.
        self.assertEqual(seen, {"feature", "other"})

        controller.observe(
            "s2",
            "Actually, ignore my earlier preference. What I need is: blue.",
            [],
        )
        # asked is cleared by the override, but candidate_attributes (and
        # therefore every score) is unchanged, so 'feature' wins again.
        after_override = controller.choose_question("s2", candidate_ids)
        self.assertEqual(after_override, "feature")


if __name__ == "__main__":
    unittest.main()
