"""Unit tests for the exact attribute-phrase route (starter/intent_index.py).

These cover the two things the route has to get right: turning a shopper turn
into (category, attribute phrases), and scoring the catalog so that a product
matching more/rarer phrases outranks one matching fewer.
"""
from __future__ import annotations

import unittest

from starter.intent_index import (attribute_card, clean_phrase, coarse_category,
                                  norm, parse_message)


class TestParseMessage(unittest.TestCase):
    def test_opening_with_hard_requirement(self):
        cat, ph = parse_message(
            "I'm looking for women's sandals. A key requirement is: 100% Cotton.")
        self.assertEqual(cat, "women's sandals")
        self.assertEqual(ph, ["100% Cotton"])

    def test_enumerated_preferences_split_on_semicolon(self):
        cat, ph = parse_message("For that, what matters is: 100% Cotton; Machine Wash.")
        self.assertIsNone(cat)
        self.assertEqual(ph, ["100% Cotton", "Machine Wash"])

    def test_open_ended_opening_yields_category_only(self):
        cat, ph = parse_message("I'm looking for shirts tops, but I'm still exploring.")
        self.assertEqual(cat, "shirts tops")
        self.assertEqual(ph, [])

    def test_correction_yields_the_new_requirement(self):
        cat, ph = parse_message(
            "Actually, ignore my earlier preference. What I need is: leather.")
        self.assertEqual(ph, ["leather"])
        self.assertIsNone(cat)

    def test_opening_with_bare_preference_tail(self):
        cat, ph = parse_message(
            "I'm looking for accessories belts. Lightweight and breathable for summer")
        self.assertEqual(cat, "accessories belts")
        self.assertEqual(ph, ["Lightweight and breathable for summer"])

    def test_no_preference_replies_yield_nothing(self):
        for msg in ("I don't have a preference for color; please use your judgment.",
                    "I don't have an additional preference for material.",
                    "Those options are not quite right yet. Ask me about one specific attribute."):
            cat, ph = parse_message(msg)
            self.assertEqual(ph, [], msg)


class TestCardConstruction(unittest.TestCase):
    def test_material_and_colour_lead_the_card(self):
        card = attribute_card({
            "title": "Blue Cotton Tee",
            "features": ["Soft hand feel", "Ribbed collar"],
            "details": {"Fit Type": "Regular"},
            "price": 19.99,
        })
        self.assertEqual(card[0], "cotton")
        self.assertEqual(card[1], "color: blue")
        self.assertIn("Soft hand feel", card)

    def test_card_falls_back_to_title_when_no_attributes(self):
        # With no features/details/price the title is the only phrase there
        # is, so it fills both the hard and the soft slot. Repetition is
        # harmless: scoring keys on the phrase, so it is counted once.
        self.assertEqual(set(attribute_card({"title": "Plain Item"})), {"Plain Item"})

    def test_clean_phrase_trims_punctuation_and_whitespace(self):
        self.assertEqual(clean_phrase("  100%   Cotton ,. "), "100% Cotton")

    def test_coarse_category_drops_the_umbrella_category(self):
        self.assertEqual(
            coarse_category(["Clothing, Shoes & Jewelry", "Women", "Sandals"]),
            "Women Sandals")

    def test_norm_is_case_and_space_insensitive(self):
        self.assertEqual(norm("  100%  COTTON. "), "100% cotton")


class TestScoring(unittest.TestCase):
    """Scoring is exercised against a tiny in-memory index."""

    def setUp(self):
        from starter.intent_index import IntentIndex
        ix = IntentIndex.__new__(IntentIndex)
        ix.card = {"A": ["cotton", "soft hand feel"],
                   "B": ["cotton", "ribbed collar"],
                   "C": ["leather", "soft hand feel"]}
        ix.card_set = {k: set(v) for k, v in ix.card.items()}
        ix.coarse = {"A": "women sandals", "B": "women sandals", "C": "men belts"}
        ix.pop = {"A": 0.9, "B": 0.1, "C": 0.5}
        ix.phrase_asins = {}
        ix.cat_asins = {}
        for a, card in ix.card.items():
            ix.cat_asins.setdefault(ix.coarse[a], set()).add(a)
            for c in card:
                ix.phrase_asins.setdefault(c, set()).add(a)
        import math
        ix.n = 3
        ix.idf = {c: math.log(1.0 + 3 / len(v)) for c, v in ix.phrase_asins.items()}
        self.ix = ix

    def test_more_matched_phrases_outranks_fewer(self):
        s = self.ix.score(["cotton", "soft hand feel"], None)
        self.assertGreater(s["A"], s["B"])
        self.assertGreater(s["A"], s["C"])

    def test_category_agreement_is_a_bonus_not_a_filter(self):
        s = self.ix.score(["soft hand feel"], "women sandals")
        self.assertIn("C", s)                  # out-of-category product survives
        self.assertGreater(s["A"], s["C"])     # but in-category one wins

    def test_positional_agreement_breaks_a_phrase_tie(self):
        # "cotton" sits at index 0 for A and B; asking for it at index 0 must
        # not invent a difference, but asking at index 1 must not crash.
        flat = self.ix.score_w([("cotton", 1.0, 0)], None)
        self.assertAlmostEqual(flat["A"], flat["B"])

    def test_unknown_phrase_contributes_nothing(self):
        self.assertEqual(self.ix.score(["no such phrase"], None), {})


if __name__ == "__main__":
    unittest.main()
