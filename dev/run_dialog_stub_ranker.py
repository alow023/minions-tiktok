"""Run the full public-set evaluator against src.dialog.DialogController's
turn policy (take_turn) using starter.stub_ranker as the sole ranker --
no real retrieval/filtering. Reports hit rate, MRR, mean turns, and the
per-scenario breakdown.

candidate_ids each turn is the full catalog minus this session's
exclude_ids: there's no LanguageFilter/RankingEngine yet to narrow it, so
stub_ranker (sort by penalty, defaulting to +inf for anything with no
recorded penalty) is doing all the "ranking." This is a smoke test of the
turn policy's mechanics (exactly 10 recs, non-null ask_attribute,
intent_override handling), not a claim about retrieval quality.

Run: python3 run_dialog_stub_ranker.py
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from src.dialog import DialogController

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10

MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
USE_CASE_RE = re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I)


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


def build_candidate_attributes(products: dict[str, dict]) -> dict[str, dict[str, str]]:
    attributes: dict[str, dict[str, str]] = {}
    for parent_asin, product in products.items():
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
    return attributes


def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    catalog_id_list = list(catalog_ids)
    candidate_attributes = build_candidate_attributes(products)

    sessions: list[dict] = []

    for sample_index, sample in enumerate(samples):
        session_id = f"run_{sample_index}"
        controller = DialogController(candidate_attributes)
        controller.reset(session_id, sample["user_profile"])

        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        scenario_type = sample["scenario_type"]

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario_type != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

        hit_turn: int | None = None
        best_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            controller.observe(session_id, user_message, [])
            state = controller.state(session_id)
            candidate_ids = [pid for pid in catalog_id_list if pid not in state["exclude_ids"]]

            turn_response = controller.take_turn(session_id, candidate_ids, top_k=TOP_K)
            ranked = [rec["parent_asin"] for rec in turn_response["recommendations"]]
            attribute = turn_response["ask_attribute"]

            assert len(ranked) == TOP_K, f"expected {TOP_K} recs, got {len(ranked)}"
            assert attribute is not None, "ask_attribute must never be null"

            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, attribute, disclosed, boundary_used
                )

        sessions.append({
            "scenario_type": scenario_type,
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"sample_count": 0, "hit_rate_at_10": 0.0, "mrr": 0.0, "mean_turns": None}
        hit_rate = sum(int(r["hit"]) for r in rows) / len(rows)
        mrr = statistics.fmean(r["reciprocal_rank"] for r in rows)
        mean_turns = statistics.fmean(
            r["first_hit_turn"] if r["first_hit_turn"] is not None else MAX_TURNS + 1 for r in rows
        )
        return {
            "sample_count": len(rows),
            "hit_rate_at_10": round(hit_rate, 6),
            "mrr": round(mrr, 6),
            "mean_turns": round(mean_turns, 4),
        }

    overall = summarize(sessions)
    print("overall:", overall)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sessions:
        grouped[row["scenario_type"]].append(row)

    print("\nby scenario_type:")
    for scenario_type in sorted(grouped):
        print(f"  {scenario_type}: {summarize(grouped[scenario_type])}")


if __name__ == "__main__":
    main()
