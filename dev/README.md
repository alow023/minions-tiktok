# dev/

Archived code from an earlier design (`src.dialog.DialogController` plus its
`legacy_starter/stub_ranker.py` and `legacy_starter/dialog_controller.py`
support). None of it is on the shipped path — the evaluator and CI only ever
import `starter.agent.Agent` at the repo root.

Kept for reference, not deleted outright, because the test suite in
`dev/tests/` documents behavior (question selection, penalty splitting,
override handling, exclusion/promotion) that this project cared about at an
earlier stage of the design.

## Running the tests here

`dev/legacy_starter/` is deliberately named differently from the root
`starter/` package so the two don't shadow each other on `sys.path`. To run
this directory's tests, put both the repo root (for `evaluator`) and `dev/`
(for `src` and `legacy_starter`) on the path:

```
PYTHONPATH=.:dev python3 -m unittest discover -s dev/tests
```

Also here: `diagnose_buckets.py` and `run_reproducibility_check.py`, which
moved out of the repo root because they reach into `evaluator.local_evaluator`
helper functions (`classify_constraint`, `intent_card`, ...) rather than
staying on the small surface (`catalog_index`, `evaluate`, `load_jsonl`) that
the shipped scripts use, and `run_dialog_stub_ranker.py`, which drives the
archived `src.dialog.DialogController` directly.
