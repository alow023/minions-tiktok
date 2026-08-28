"""Runs either the baseline (starter/agent.py) or this module's agent
(starter/fake_agent.py) through the exact same evaluator.local_evaluator
scoring path, and prints a side-by-side comparison against
docs/baseline_results.json.

Does not modify evaluator/local_evaluator.py in any way -- only imports
load_jsonl, catalog_index, and evaluate from it.

Run:
    python3 run_eval.py                       # this module's agent (fake)
    python3 run_eval.py --agent starter        # the untouched baseline
    python3 run_eval.py --catalog ... --dataset ... --output ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

BASELINE_RESULTS_PATH = "docs/baseline_results.json"


def _load_agent_class(name: str):
    if name == "starter":
        from starter.agent import Agent
        return Agent
    if name == "fake":
        from starter.fake_agent import Agent
        return Agent
    raise ValueError(f"unknown --agent {name!r}")


def _print_baseline_comparison(result: dict) -> None:
    baseline_path = Path(BASELINE_RESULTS_PATH)
    if not baseline_path.exists():
        print(f"\n(no baseline comparison: {BASELINE_RESULTS_PATH} not found)")
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    rows = [
        ("hit_rate_at_10", baseline.get("hit_rate_at_10"), result.get("hit_rate_at_10")),
        ("mrr", baseline.get("mrr"), result.get("mrr")),
        ("mttc", baseline.get("mttc"), result.get("mttc")),
        ("technical_score", baseline.get("technical_score"), result.get("recommended_technical_score")),
    ]

    print(f"\nComparison vs baseline ({baseline.get('baseline', '?')}, {BASELINE_RESULTS_PATH}):")
    header = f"  {'metric':<16} {'baseline':>10} {'this run':>10} {'delta':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, base_val, this_val in rows:
        if base_val is None or this_val is None:
            print(f"  {name:<16} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
            continue
        delta = this_val - base_val
        note = ""
        if name == "mttc":
            note = "  (lower is better)"
        print(f"  {name:<16} {base_val:>10.6f} {this_val:>10.6f} {delta:>+10.6f}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline or fake agent through the local evaluator")
    parser.add_argument("--agent", choices=["starter", "fake"], default="fake", help="which Agent to evaluate")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    AgentClass = _load_agent_class(args.agent)

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    result = evaluate(AgentClass(args.catalog), samples, catalog_ids, categories, products)

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, indent=2))

    _print_baseline_comparison(result)


if __name__ == "__main__":
    main()
