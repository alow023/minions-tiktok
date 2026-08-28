"""Stub ranking function — ordering only, no retrieval logic yet."""

from __future__ import annotations

import math


def rank(candidate_ids: list[str], state: dict, k: int = 10) -> list[str]:
    """Return up to k candidate ids not in state['exclude_ids'], ordered by
    ascending penalty from state['penalties'] (missing penalty sorts last).
    """
    exclude_ids = state.get("exclude_ids") or set()
    penalties = state.get("penalties") or {}

    eligible = [cid for cid in candidate_ids if cid not in exclude_ids]
    ordered = sorted(eligible, key=lambda cid: penalties.get(cid, math.inf))
    return ordered[:k]
