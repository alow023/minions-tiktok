"""Runs the full 200-session public set through starter.fake_agent.Agent
and dumps a deterministic, ordered per-session results file. Meant to be
run twice (as two separate `python3` invocations, so any PYTHONHASHSEED-
dependent nondeterminism actually shows up) and diffed byte-for-byte.

Run: python3 run_reproducibility_check.py <output.json>
"""

from __future__ import annotations

import json
import sys

from evaluator.local_evaluator import (
    MAX_TURNS,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.fake_agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "reproducibility_run.json"

    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    agent = Agent(CATALOG_PATH)

    results = []

    for sample in samples:
        sample_id = sample["sample_id"]
        session_id = f"repro_{sample_id}"
        scenario_type = sample["scenario_type"]

        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario_type != "intent_override"
        user_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed
        )

        turns = []
        hit_turn = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            ranked = [r["parent_asin"] for r in response["recommendations"]]
            state = agent.controller.state(session_id)

            # Full penalties dicts can hold tens of thousands of entries by
            # later turns (see diagnose_run.py's findings), so dumping them
            # whole is impractical for a byte-diff. A bounded, order-
            # sensitive sample is enough to expose the exact bug this
            # script exists to catch: state()'s key ORDER used to depend
            # on a set union (PYTHONHASHSEED-randomized per process); the
            # first N keys in dict order reveal that just as well as the
            # full dict would, at a fraction of the size.
            penalty_items = list(state["penalties"].items())
            turns.append({
                "turn": turn,
                "ask_attribute": response["ask_attribute"],
                "recommendations": ranked,
                "penalty_count": len(penalty_items),
                "penalty_key_order_sample": [pid for pid, _ in penalty_items[:20]],
                "penalty_value_sample": [round(v, 6) for _, v in penalty_items[:20]],
                "constraint_count": len(state["constraints"]),
            })

            if override_applied and target in ranked:
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
                    effective_sample, response["ask_attribute"], disclosed, boundary_used
                )

        results.append({
            "sample_id": sample_id,
            "scenario_type": scenario_type,
            "hit_turn": hit_turn,
            "turns": turns,
        })

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
