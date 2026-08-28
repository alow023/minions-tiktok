"""Smoke-test the evaluator loop against starter.agent.Agent.

Originally exercised a placeholder (starter/fake_agent.py, now removed);
kept as a quick end-to-end sanity check that reset/respond/turn-looping
still works against the real agent. For an actual scored run with the
baseline-comparison table, use run_eval.py instead.

Run: python3 run_fake_agent.py
"""

from __future__ import annotations

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"


def main() -> None:
    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)
    result = evaluate(Agent(CATALOG_PATH), samples, catalog_ids, categories, products)
    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(summary)


if __name__ == "__main__":
    main()
