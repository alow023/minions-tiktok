"""One-command reproducibility and latency disclosure for starter.agent.Agent.

Default mode (no arguments) is the driver: it launches two separate
`python3` subprocesses in a fully scrubbed environment (`env -i`, plus only
the PATH and PYTHONPATH needed to find python3 and this repo's packages --
no PYTHONHASHSEED, no leftover C_* / GATE / QPOLICY / RERANKER config from
the calling shell), each running the full 200-session public set through
Agent and dumping a deterministic, ordered per-session results file. It then
hashes both dump files and prints IDENTICAL or DIFFERENT, along with the
technical score from each run and the latency/index-build figures the
submission rules require as a disclosure.

Two subprocesses rather than two in-process runs because that's the only
way PYTHONHASHSEED-dependent nondeterminism (str/bytes hashing affecting
set/dict iteration order -- Agent builds several sets during indexing:
self.vocab values, gaz = set().union(*self.vocab.values()), per-product
term sets) would actually show up; two calls to the same function in one
process share a hash seed and can't catch that class of bug.

Run: python3 dev/run_reproducibility_check.py
     python3 dev/run_reproducibility_check.py --worker <output.json>   (internal)
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = "data/catalog.jsonl"
DATASET_PATH = "data/public_set.jsonl"
TOP_K = 10


def _worker(output_path: str) -> None:
    """Single full run: build the index (timed), run all 200 sessions
    (each agent.respond() call timed), write a deterministic ordered dump,
    and print one JSON summary line to stdout for the driver to parse."""
    from evaluator.local_evaluator import (
        MAX_TURNS,
        catalog_index,
        coarse_category,
        customer_reply,
        evaluate,
        initial_message,
        load_jsonl,
        materialize_hidden_fields,
    )
    from starter.agent import Agent

    samples = load_jsonl(DATASET_PATH)
    catalog_ids, categories, products = catalog_index(CATALOG_PATH)

    index_start = time.perf_counter()
    agent = Agent(CATALOG_PATH)
    index_build_seconds = time.perf_counter() - index_start

    latencies_ms: list[float] = []
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
            call_start = time.perf_counter()
            response = agent.respond(session_id, user_message, turn, TOP_K)
            latencies_ms.append((time.perf_counter() - call_start) * 1000.0)

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
                "shown_count": len(state.get("shown", ())),
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

    # Re-run the same 200 samples through the standard scoring path for the
    # technical score. A fresh Agent so this doesn't share state with the
    # per-session dump above.
    score_result = evaluate(Agent(CATALOG_PATH), samples, catalog_ids, categories, products)

    latencies_ms.sort()
    n = len(latencies_ms)
    median_ms = latencies_ms[n // 2] if n else 0.0
    p95_ms = latencies_ms[min(n - 1, int(n * 0.95))] if n else 0.0

    summary = {
        "recommended_technical_score": score_result["recommended_technical_score"],
        "index_build_seconds": index_build_seconds,
        "turn_count": n,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
    }
    print("REPRO_SUMMARY " + json.dumps(summary))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_worker_subprocess(output_path: str) -> dict:
    scrubbed_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", output_path],
        cwd=str(REPO_ROOT),
        env=scrubbed_env,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("REPRO_SUMMARY "):
            return json.loads(line[len("REPRO_SUMMARY "):])
    raise RuntimeError(f"worker produced no summary line; stdout was:\n{proc.stdout}\nstderr:\n{proc.stderr}")


def _driver() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = str(Path(tmpdir) / "run1.json")
        out2 = str(Path(tmpdir) / "run2.json")

        print("Running full evaluation twice in a scrubbed environment (env -i)...")
        summary1 = _run_worker_subprocess(out1)
        summary2 = _run_worker_subprocess(out2)

        hash1 = _sha256(out1)
        hash2 = _sha256(out2)

        verdict = "IDENTICAL" if hash1 == hash2 else "DIFFERENT"
        print(f"\n{verdict}")
        print(f"  run 1: score={summary1['recommended_technical_score']:.6f}  sha256={hash1}")
        print(f"  run 2: score={summary2['recommended_technical_score']:.6f}  sha256={hash2}")

        print("\nLatency disclosure (run 1, per-turn agent.respond() calls, "
              f"n={summary1['turn_count']}):")
        print(f"  index build time: {summary1['index_build_seconds']:.4f} s")
        print(f"  median latency:   {summary1['median_latency_ms']:.3f} ms")
        print(f"  p95 latency:      {summary1['p95_latency_ms']:.3f} ms")

        if verdict != "IDENTICAL":
            sys.exit(1)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
        return
    _driver()


if __name__ == "__main__":
    main()
