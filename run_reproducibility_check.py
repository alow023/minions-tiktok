"""Runs the full 200-session public set through starter.agent.Agent and
dumps a deterministic, ordered per-session results file. Meant to be run
twice (as two separate `python3` invocations, so any PYTHONHASHSEED-
dependent nondeterminism actually shows up) and diffed byte-for-byte.

starter.agent.Agent builds several sets during indexing (self.vocab values,
gaz = set().union(*self.vocab.values()), per-product term sets intersected
with that gazetteer) and tracks per-session state as Counters/sets
(bad_values, shown, asked) -- none of that is inherently guaranteed
order-stable across process runs just because it "looks like a dict", so
this is worth checking directly rather than assuming it's fine.

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
from starter.agent import Agent

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

            state = agent.state[session_id]
            bad_values = sorted(
                f"{attribute}:{value}={count}"
                for (attribute, value), count in state["bad_values"].items()
            )

            turns.append({
                "turn": turn,
                "ask_attribute": response["ask_attribute"],
                "recommendations": ranked,
                "shown_count": len(state["shown"]),
                "bad_values_sample": bad_values[:20],
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
