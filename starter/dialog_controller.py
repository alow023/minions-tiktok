"""Empty DialogController stub — plumbing only, no real conversation logic.

Swap the body of `step()` out for real NLU/state-tracking once the loop is
verified end to end.
"""

from __future__ import annotations


class DialogController:
    def __init__(self, catalog_ids: list[str]) -> None:
        self.catalog_ids = catalog_ids
        self.sessions: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {"exclude_ids": set(), "penalties": {}, "turn": 0}

    def step(self, session_id: str, user_message: str, turn: int) -> tuple[str, str | None, list[str], dict]:
        """Return (message, ask_attribute, candidate_ids, state). Stub: no
        parsing of user_message, no real candidate selection or penalties.
        """
        state = self.sessions[session_id]
        state["turn"] = turn
        message = "stub: no dialog logic yet"
        ask_attribute = "other"
        candidate_ids = self.catalog_ids
        return message, ask_attribute, candidate_ids, state
