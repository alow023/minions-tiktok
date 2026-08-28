"""Agent wired to the REAL DialogController (src.dialog) via take_turn(),
which itself ranks with starter.stub_ranker. Used for local diagnostics and
invariant testing -- NOT the actual submission entry point (that remains
starter/agent.py, untouched).

Includes a deliberately lightweight constraint extractor (not a real
LanguageFilter): just enough message-template parsing to hand
DialogController.observe() meaningful {'text','attribute','kind'} dicts
instead of an empty list, using the evaluator's own classify_constraint()
so attribute labels match exactly what the simulator itself would assign.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluator.local_evaluator import classify_constraint
from src.dialog import DialogController

MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
USE_CASE_RE = re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I)

BUYING_RE = re.compile(r"^I'm looking for .+?\. A key requirement is: (?P<constraint>.+)\.$")
OVERRIDE_RE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (?P<constraint>.+)\.$")
MATCHES_RE = re.compile(r"^For that, what matters is: (?P<body>.+)\.$")
OVERRIDE_INITIAL_RE = re.compile(r"^I'm looking for .+?\. (?P<constraint>.+)$")


def _searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _price_bucket(price: object) -> str | None:
    if price in (None, ""):
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price < 15:
        return "under_15"
    if price < 30:
        return "15_30"
    if price < 60:
        return "30_60"
    return "over_60"


def _build_candidate_attributes(catalog_path: str | Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    attributes: dict[str, dict[str, str]] = {}
    ids: list[str] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            ids.append(parent_asin)
            text = _searchable_text(product)
            entry: dict[str, str] = {}
            material = MATERIAL_RE.search(text)
            color = COLOR_RE.search(text)
            use_case = USE_CASE_RE.search(text)
            if material:
                entry["material"] = material.group(1).lower()
            if color:
                entry["color"] = color.group(1).lower()
            if use_case:
                entry["use_case"] = use_case.group(1).lower()
            budget = _price_bucket(product.get("price"))
            if budget:
                entry["budget"] = budget
            details = product.get("details")
            if isinstance(details, dict):
                department = details.get("Department")
                if department:
                    entry["style"] = str(department).strip().lower()
            attributes[parent_asin] = entry
    return attributes, ids


def extract_constraints(message: str) -> list[dict]:
    """Best-effort {'text','attribute','kind'} extraction from the
    customer's message text. Not a real LanguageFilter -- just enough to
    exercise DialogController.observe() with meaningful constraint data.
    """
    text = (message or "").strip()

    match = BUYING_RE.match(text)
    if match:
        value = match.group("constraint")
        return [{"text": value, "attribute": classify_constraint(value), "kind": "hard"}]

    match = OVERRIDE_RE.match(text)
    if match:
        value = match.group("constraint")
        return [{"text": value, "attribute": classify_constraint(value), "kind": "hard"}]

    match = MATCHES_RE.match(text)
    if match:
        constraints = []
        for chunk in match.group("body").split("; "):
            chunk = chunk.strip()
            if chunk:
                constraints.append({"text": chunk, "attribute": classify_constraint(chunk), "kind": "soft"})
        return constraints

    if "please use your judgment" in text or "additional preference" in text:
        return []

    if text.startswith("I'm looking for ") and "but I'm still exploring" not in text:
        match = OVERRIDE_INITIAL_RE.match(text)
        if match:
            value = match.group("constraint")
            return [{"text": value, "attribute": classify_constraint(value), "kind": "soft"}]

    return []


class Agent:
    """Wires the real DialogController to stub_ranker via take_turn()."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        candidate_attributes, catalog_ids = _build_candidate_attributes(catalog_path)
        self.controller = DialogController(candidate_attributes)
        self._catalog_ids = catalog_ids

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.controller.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        constraints = extract_constraints(user_message)
        self.controller.observe(session_id, user_message, constraints)

        state = self.controller.state(session_id)
        candidate_ids = [pid for pid in self._catalog_ids if pid not in state["exclude_ids"]]

        turn_result = self.controller.take_turn(session_id, candidate_ids, top_k=top_k)
        return {
            "message": "Here's what I found so far -- let me know if anything else matters.",
            "ask_attribute": turn_result["ask_attribute"],
            "recommendations": turn_result["recommendations"],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
