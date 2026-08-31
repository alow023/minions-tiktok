"""Diagnostic: empirical distribution of classify_constraint() buckets over
intent_card() output, sampled from the real catalog.

Samples 2000 random products from data/catalog.jsonl, builds each one's
intent_card(), and classifies every resulting hard_constraint and
soft_preference string with classify_constraint(). Prints the bucket
frequency distribution overall, and separately for hard_constraints vs
soft_preferences.

This is what src/dialog.py's EMPIRICAL_ATTRIBUTE_PRIORS are derived from
(see the comment there). Re-run this and update that constant if the
catalog or intent_card()/classify_constraint() ever change. The default
--seed (0) reproduces the exact values already recorded there; pass a
different --seed to sanity-check how much the distribution moves across
samples.

Run: python3 diagnose_buckets.py [--seed N] [--sample-size N]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from evaluator.local_evaluator import classify_constraint, intent_card

CATALOG_PATH = "data/catalog.jsonl"
SAMPLE_SIZE = 2000
SEED = 0


def load_catalog(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _print_distribution(title: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n{title} (n={total})")
    if total == 0:
        print("  (no values)")
        return
    for bucket, count in counts.most_common():
        print(f"  {bucket:10s} {count:5d}  ({100 * count / total:5.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--seed", type=int, default=SEED, help="random seed for product sampling (default: 0)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, dest="sample_size")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    rng = random.Random(args.seed)
    sample = rng.sample(catalog, min(args.sample_size, len(catalog)))

    hard_counts: Counter = Counter()
    soft_counts: Counter = Counter()

    for product in sample:
        card = intent_card(product)
        for value in card.get("hard_constraints", []):
            hard_counts[classify_constraint(str(value))] += 1
        for value in card.get("soft_preferences", []):
            soft_counts[classify_constraint(str(value))] += 1

    overall_counts = hard_counts + soft_counts

    _print_distribution(f"overall distribution across {len(sample)} sampled products", overall_counts)
    _print_distribution("hard_constraints only", hard_counts)
    _print_distribution("soft_preferences only", soft_counts)


if __name__ == "__main__":
    main()
