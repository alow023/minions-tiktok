"""Parallel wrapper around the official evaluator (identical metrics)."""
import json, os, sys, statistics
from collections import defaultdict
from multiprocessing import Pool
from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index, metric_summary, MAX_TURNS)
from starter.agent import Agent

CAT = "data/catalog.jsonl"
_G = {}

def init():
    ids, cats, prods = catalog_index(CAT)
    _G["a"] = Agent(CAT); _G["i"] = ids; _G["c"] = cats; _G["p"] = prods

def work(chunk):
    return evaluate(_G["a"], chunk, _G["i"], _G["c"], _G["p"])["sessions"]

if __name__ == "__main__":
    samples = load_jsonl("data/public_set.jsonl")
    n = int(os.environ.get("NPROC", "8"))
    chunks = [samples[i::n] for i in range(n)]
    with Pool(n, initializer=init) as pool:
        sessions = [s for r in pool.map(work, chunks) for s in r]
    overall = metric_summary(sessions)
    eff = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    score = 0.50*overall["hit_rate_at_10"] + 0.30*overall["mrr"] + 0.20*eff
    g = defaultdict(list)
    for s in sessions: g[s["scenario_type"]].append(s)
    out = {**overall, "efficiency": round(eff,6), "recommended_technical_score": round(score,6),
           "scenario_metrics": {k: metric_summary(g[k]) for k in sorted(g)}}
    print(json.dumps(out, indent=1))
    json.dump({**out, "sessions": sessions}, open(sys.argv[1] if len(sys.argv)>1 else "res.json","w"), indent=1)
