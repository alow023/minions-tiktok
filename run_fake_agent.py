"""Smoke-test the evaluator loop against fake_agent.Agent (empty DialogController stub).

Verifies the reset/respond plumbing and turn loop work before any real
retrieval/dialog logic is written. Does not import or modify starter.agent.

Run: python3 run_fake_agent.py
"""

from __future__ import annotations

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.fake_agent import Agent

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
