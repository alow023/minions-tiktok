# dev/

Development-time tooling. Nothing here is on the shipped path — the evaluator
and CI only ever import `starter.agent.Agent`.

- `robustness/paraphrase_eval.py` — rewrites the simulator's utterances at four
  severities (`none`/`light`/`medium`/`heavy`) *without* modifying the
  evaluator, and re-scores the shipped agent against them. Results in
  `robustness/res_*.json`; see `docs/ABLATION.md` §3.6 and §3.10.
  Run: `python3 -m dev.robustness.paraphrase_eval heavy`
- `robustness/diagnose.py` — per-session diagnostics for a scored run.
- `run_eval_parallel.py` — multiprocess wrapper around the official evaluator's
  own `evaluate`/`metric_summary` functions. Identical metrics, much faster on
  a multi-core box. Report scores from `python3 -m evaluator.local_evaluator`.
  Run: `NPROC=8 python3 dev/run_eval_parallel.py results.json`
- `run_reproducibility_check.py` — re-runs the evaluator and checks the score
  is bit-identical across runs.

The archived pre-BM25 dialog stack (`src/dialog.py`, `legacy_starter/`, and the
30 unit tests that exercised them) was removed once the shipped architecture
had fully superseded it; it is recoverable from git history at `f7e7dba`
onwards. Its useful behavioural coverage lives on in `tests/`.
