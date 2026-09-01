# Conversational Shopping Agent — TechJam Conversational Search

A multi-turn shopping agent for the TechJam conversational-search challenge. It
converges on the customer's target product out of a frozen 50,000-product
Amazon Clothing/Shoes/Jewelry catalog, in as few turns as possible, using no
external model API, no network access at runtime, and no pretrained weights.

**Public-set result (200 sessions, official evaluator, unmodified):**

| | Hit@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---|---|---|---|---|
| Provided weak BM25 baseline | 0.125 | 0.0680 | 9.81 | 0.119 | 0.10671 |
| **This agent** | **0.995** | **0.8647** | **2.805** | **0.8195** | **0.920808** |

Per scenario: buying MRR 0.900 / MTTC 2.23 · browsing 0.837 / 2.80 ·
intent_override 0.814 / 4.17 · boundary 0.950 / 3.40. See results.json for the
exact per-scenario table.

---

## How it addresses the problem statement

**I. Intent routing and a hybrid pipeline.** Every turn runs a multi-route
in-memory retrieval fused by reciprocal rank, then re-ranked:

| route | file | signal |
|---|---|---|
| Field-weighted BM25 (unigrams + bigrams) | `starter/agent.py` (`BM25`) | lexical |
| Category overlap + soft constraint boosts | `starter/agent.py` (`_allowed_set`, `_fuse`) | structured facets |
| Catalog-derived LSA (TF-IDF → truncated SVD) | `starter/semantic.py` | dense/semantic |
| **Attribute-phrase reranking** | `starter/intent_index.py` | verbatim catalog attribute values |

The *high-precision* track is the phrase route: a stated hard constraint is
matched as a whole canonical attribute phrase, not a bag of words. The
*diverse* track is BM25 + LSA, which carries open-ended browsing turns where no
constraint has been stated yet. The two are fused with a confidence-scaled
weight (`Agent._fp_blend`) — lexical leads while the customer is still
exploring, exact phrases take over the moment a requirement lands.

**II. Multi-turn state.** `Agent`/`ShoppingCopilot` keep a per-session state
machine: slots accumulate incrementally and order-preservingly, an override cue
(`_is_override`) rewrites them — stale constraints dropped, shown-set, penalty
counters and asked-attribute set reset — while pre-override turns are retained
at a reduced weight, because an override rewrites a *preference*, not the
shopping mission. Over-general turns are handled by `_gate_count`: when the
candidate pool has no clear leader the agent withholds the slate and asks a
clarification instead of dumping ten guesses.

**III. Runtime adaptation.** `_learn_from_rejection` distils a rejected slate
into negative evidence (`bad_values`) that penalises repeat attributes;
`_profile_prior` folds the session's anonymised profile into early-turn
ranking; `_fp_blend` re-weights the whole pipeline per turn from the phrase
route's own confidence, so the workflow re-orchestrates itself as the dialog
sharpens.

**IV. Efficiency.** Slate width is an explicit decision, not a constant. MRR is
scored at the *first* turn the target surfaces, so widening the slate converts
sooner but at a worse rank — measured at 7:1 against, and the agent is tuned to
that trade-off (`docs/ABLATION.md` §2, `C_FP_SLATE`).

### The core idea, in one paragraph

Shoppers in this domain don't paraphrase requirements, they **quote** them:
"100% Cotton", "Machine Wash", "color: black". Those strings are the catalog's
own canonical attribute values. BM25 over the same fields discards the phrase
boundaries. So we build, from `catalog.jsonl` alone, one canonical attribute
card per product and an inverted index from phrase → products. On the public
set this route never ranks the target below the top evidence score and never
drops it from the pool — **every** residual error is a tie between colourway or
size variants of one listing family that share a feature card verbatim. Those
ties are broken by review volume (the listing people actually buy), applied
scaled by the route's own confidence so it is inert while evidence is thin.
That single mechanism is worth **+0.023** technical score
(`docs/ABLATION.md` §3.8). It reranks only within the lexical candidate pool and
never injects a product BM25/LSA did not surface, so a wrong phrase match can
cost rank but never recall — removing that bound was measured at 0.879667,
*below* the lexical-only baseline.

## Setup

Python 3.10+.

```bash
gzip -dk data/catalog.jsonl.gz          # produces data/catalog.jsonl (50,000 rows)
pip install -r requirements.txt         # scikit-learn only; optional
```

`scikit-learn` powers the LSA route. If it is absent the agent catches the
ImportError and degrades to the pure-standard-library pipeline, scoring
`0.919` instead of `0.920808`. Nothing else is required — no API keys, no
network, no downloaded model weights.

## Reproduce the result

```bash
env -i PATH="$PATH" python3 -m evaluator.local_evaluator --output results.json
```

Runs the 200 public sessions through the unmodified official evaluator and
writes per-session results plus aggregates. Expect
`recommended_technical_score = 0.920808`. Every tunable is an environment
variable with a measured default; the scrubbed environment above guarantees you
get the defaults. `dev/run_eval_parallel.py` is a multiprocess wrapper around
the same evaluator functions (identical metrics, faster on multi-core boxes).

```bash
python3 -m unittest discover -s tests            # unit + full-run invariants
python3 -m dev.robustness.paraphrase_eval heavy  # paraphrase-robustness harness
```

Determinism: no sampling, no network, no clock or PID dependence; the LSA route
is fitted with a fixed `random_state`. Reported token usage is 0/0 — there is
no LLM in the loop.

## Repository map

```text
starter/agent.py          the agent: BM25 base + ShoppingCopilot pipeline, state machine
starter/intent_index.py   attribute-phrase reranking route (inverted phrase index)
starter/semantic.py       catalog-derived LSA route (scikit-learn, optional)
evaluator/                official evaluator, unmodified
tests/                    unit tests + whole-public-set invariant checks
dev/robustness/           paraphrase-robustness harness and diagnostics
dev/run_eval_parallel.py  multiprocess wrapper around the official evaluator
docs/CONFIGURATION.md     every environment variable, its default, and its measured delta
docs/ABLATION.md          leave-one-out ablations, instrumentation, validation methodology
docs/challenge_readme.md  the organizer's original README (challenge rules and API contract)
```

## Development tools, libraries, data

- **Tools:** VS Code, git/GitHub Actions (`.github/workflows/score.yml` re-scores
  every push and fails on regression), Python 3.10/3.11, `unittest`.
- **Libraries:** Python standard library for everything on the critical path;
  `scikit-learn` (`TfidfVectorizer`, `TruncatedSVD`) for the optional LSA route.
- **APIs / models:** none. No LLM API, no hosted service, no pretrained weights.
- **Data:** the organizer's frozen `catalog.jsonl` (50,000 Amazon Reviews 2023
  Clothing/Shoes/Jewelry products) and `data/public_set.jsonl` (200 labeled
  sessions), both read-only and unmodified. See `DATA_ATTRIBUTION.md`.

## Team contributions

| Team member | Contributions                                                                                                                                                                                      |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Claire**  | Developed the semantic retrieval component using TF-IDF, Latent Semantic Analysis, and cosine-similarity reranking. Helped integrate the semantic route with the existing BM25 candidate pipeline. |
| **Xiaomei** | Developed the dialogue-state and intent-routing logic, including buyer-versus-explorer classification, constraint extraction, preference overrides, and adaptive clarification questions.          |
| **Arissa**  | Developed the multi-route ranking and evaluation pipeline, including Reciprocal Rank Fusion, confidence gating, candidate rotation, metric analysis, and configuration tuning.                     |

All three members contributed to solution design, debugging, evaluator testing, performance comparison, documentation, and preparation of the final presentation.

## Limitations and what we would improve

- **Phrase-route coverage depends on catalog metadata quality.** Products with
  thin or boilerplate `features`/`details` produce weak cards, and their listing
  families tie. We break those ties with review volume; a stronger answer is a
  learned variant-family model that scores which member of a family a given
  customer buys.
- **The parser is rule-based.** It handles the natural shapes of shopper speech
  and falls through to the lexical pipeline on anything else, so it is safe but
  not exhaustive. A small local intent/slot model would widen coverage without
  adding an API dependency.
- **`intent_override` and `boundary` sit near structural floors** (MTTC 3.63 and
  2.90 against ~3.5 and ~3.0). The remaining headroom is almost entirely in
  browsing turn 1, where only a category is known.
- **No cross-session learning.** Long-term user profiles are used as a ranking
  prior only; a persistent per-user model across sessions is the natural next
  step and is what the "self-evolution" pillar would reward at production scale.
- **Single-process, in-memory.** Index build is ~30s for 50k products and the
  whole index fits in RAM, which is the point for this challenge, but a
  production deployment would want an incremental index and sharding.
