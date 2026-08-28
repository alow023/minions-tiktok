"""Robustness edge cases for DialogController. None of these should ever
raise, and choose_question must always return a legal attribute string
regardless of how malformed or degenerate the surrounding calls are.
"""

from __future__ import annotations

import unittest

from src.dialog import ASKABLE_ATTRIBUTES, LEGAL_ATTRIBUTES, DialogController

PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "x",
}


class EdgeCasesTest(unittest.TestCase):
    def assert_legal_attribute(self, controller: DialogController, session_id: str, candidate_ids=None) -> None:
        result = controller.choose_question(session_id, candidate_ids if candidate_ids is not None else [])
        self.assertIsNotNone(result)
        self.assertIn(result, LEGAL_ATTRIBUTES)

    def test_observe_on_never_reset_session(self) -> None:
        controller = DialogController()
        try:
            controller.observe("never-reset", "I'm looking for Shirts, but I'm still exploring.", [])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"observe() raised on a never-reset session: {exc!r}")
        self.assert_legal_attribute(controller, "never-reset")

    def test_reset_twice_fully_clears_prior_state(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)
        controller.observe("s1", "I'm looking for Shirts. A key requirement is: cotton.", [
            {"text": "cotton", "attribute": "material", "kind": "hard"}
        ])
        controller.register_slate("s1", ["P1", "P2"])
        controller.observe("s1", "For that, what matters is: something.", [])

        state_before = controller.state("s1")
        self.assertTrue(state_before["constraints"])
        self.assertTrue(state_before["exclude_ids"] or state_before["penalties"])
        self.assertGreater(state_before["turn"], 0)

        controller.reset("s1", PROFILE)
        state_after = controller.state("s1")
        self.assertEqual(state_after["constraints"], [])
        self.assertEqual(state_after["exclude_ids"], set())
        self.assertEqual(state_after["penalties"], {})
        self.assertEqual(state_after["turn"], 0)
        self.assert_legal_attribute(controller, "s1")

    def test_register_slate_with_empty_list(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)
        try:
            controller.register_slate("s1", [])
            controller.observe("s1", "For that, what matters is: something.", [])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"empty register_slate() raised: {exc!r}")
        self.assert_legal_attribute(controller, "s1")

    def test_choose_question_with_empty_candidate_ids(self) -> None:
        controller = DialogController({"P1": {"color": "blue"}})
        controller.reset("s1", PROFILE)
        try:
            result = controller.choose_question("s1", [])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"choose_question() raised on empty candidate_ids: {exc!r}")
        self.assertIn(result, LEGAL_ATTRIBUTES)

    def test_choose_question_when_everything_asked_or_exhausted(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)
        session = controller.sessions["s1"]
        # Split the 7 askable buckets between asked and exhausted so both
        # skip-conditions in choose_question's loop are exercised.
        askable = list(ASKABLE_ATTRIBUTES)
        session["asked"] = set(askable[: len(askable) // 2])
        session["exhausted"] = set(askable[len(askable) // 2 :])
        try:
            result = controller.choose_question("s1", ["P1", "P2"])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"choose_question() raised when everything is asked/exhausted: {exc!r}")
        self.assertEqual(result, "other")
        self.assertIn(result, LEGAL_ATTRIBUTES)

    def test_full_session_with_broken_upstream_parser(self) -> None:
        # extract_constraints is simulated as always returning [] -- a
        # completely broken upstream LanguageFilter. observe() must still
        # survive 10 full turns and choose_question must keep answering.
        candidate_attributes = {f"P{i}": {"color": "blue" if i % 2 else "red"} for i in range(30)}
        controller = DialogController(candidate_attributes)
        controller.reset("s1", PROFILE)

        messages = [
            "I'm looking for Shirts, but I'm still exploring.",
            "For that, what matters is: something.",
            "Actually, ignore my earlier preference. What I need is: something else.",
            "I don't have an additional preference for other.",
            "I don't have a preference for material; please use your judgment.",
        ]

        try:
            for turn in range(1, 11):
                message = messages[turn % len(messages)]
                controller.observe("s1", message, [])  # always empty constraints
                candidate_ids = [pid for pid in candidate_attributes if pid not in controller.state("s1")["exclude_ids"]]
                attribute = controller.choose_question("s1", candidate_ids)
                self.assertIn(attribute, LEGAL_ATTRIBUTES)
                controller.register_slate("s1", candidate_ids[:10])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"10-turn session with a broken constraint parser raised: {exc!r}")

        self.assertEqual(controller.state("s1")["constraints"], [])
        self.assert_legal_attribute(controller, "s1", list(candidate_attributes.keys()))

    def test_message_with_unicode_emoji_and_very_long_text(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)
        message = (
            "I'm looking for \U0001f9e6 shirts \U0001f680\U0001f680\U0001f680 "
            "日本語のテキストです。これはテストです。"
            + ("x" * 50_000)
            + ", but I'm still exploring."
        )
        try:
            controller.observe("s1", message, [])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"observe() raised on unicode/emoji/very-long message: {exc!r}")
        self.assert_legal_attribute(controller, "s1")

    def test_register_slate_called_with_same_ids_twice_in_a_row(self) -> None:
        controller = DialogController()
        controller.reset("s1", PROFILE)
        try:
            controller.register_slate("s1", ["A", "B"])
            controller.register_slate("s1", ["A", "B"])
            controller.observe("s1", "For that, what matters is: something.", [])
        except Exception as exc:  # noqa: BLE001
            self.fail(f"register_slate() called twice with the same ids raised: {exc!r}")
        # Only one promotion happened (the second call replaced the
        # pending slate, it didn't queue a second one) -- A/B end up
        # excluded exactly once, not double-counted anywhere.
        self.assertEqual(controller.state("s1")["exclude_ids"], {"A", "B"})
        self.assert_legal_attribute(controller, "s1")

    def test_state_on_unknown_session_id(self) -> None:
        controller = DialogController()
        try:
            state = controller.state("never-heard-of-it")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"state() raised on an unknown session id: {exc!r}")
        self.assertEqual(state["constraints"], [])
        self.assertEqual(state["exclude_ids"], set())
        self.assertEqual(state["penalties"], {})
        self.assertEqual(state["turn"], 0)
        self.assert_legal_attribute(controller, "never-heard-of-it")


if __name__ == "__main__":
    unittest.main()
