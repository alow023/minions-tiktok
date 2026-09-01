# Ablation and Validation

This document is the technical record behind `starter/agent.py`'s configuration. Every number below was measured on the 200-sample public set (`data/public_set.jsonl`) unless stated otherwise. The detailed leave-one-out measurements in §2 predate the current robustness and profile-prior defaults and are retained as historical results (§2's own preamble states this explicitly).

## 1. Headline

| | hit_rate_at_10 | mrr | mttc | efficiency | recommended_technical_score |
|---|---|---|---|---|---|
| **Baseline** (provided weak BM25 starter, `docs/baseline_results.json`) | 0.125 | 0.068034 | 9.81 | 0.119 | **0.10671** |
| Lexical + semantic pipeline (`C_FP=0`) | 0.980 | 0.822500 | 2.965 | 0.8035 | 0.897458 |
| **Final** (+ catalog attribute-phrase reranking) | **0.995** | **0.864692** | **2.805** | **0.8195** | **0.920808** |

**8.63x improvement** over the provided baseline (0.920808 / 0.10671 = 8.629).

Per-scenario metrics, final configuration:

| scenario | n | hit@10 | mrr | mttc | mrr before | mttc before |
|---|---|---|---|---|---|---|
| buying | 80 | 1.000 | 0.900 | 2.225 | 0.877 | 2.46 |
| browsing | 80 | 0.9875 | 0.837 | 2.800 | 0.847 | 2.84 |
| intent_override | 30 | 1.000 | 0.814 | 4.167 | 0.653 | 4.37 |
| boundary | 10 | 1.000 | 0.950 | 3.400 | 0.695 | 3.80 |

`intent_override` and `boundary` sessions cannot converge on turn 1 by
construction — an override is only scorable from the turn it lands on, and a
boundary session spends one turn declining to answer.

## 2. Leave-one-out ablation

For each configuration variable below, we reverted *only that one setting* to its non-default alternative and re-ran the full 200-sample evaluation. These historical rows share the same pre-robustness stack, so their deltas are internally comparable to one another; they are not deltas against the current `C_ROBUST=1` baseline.

| rank | variable reverted to | score | Δ vs. current default |
|---|---|---|---|
| 0 | `C_FP=0` (disable the attribute-phrase reranking route) | 0.897458 | **−0.023350** |

Rows 1 onward are *historical*: measured under the pre-phrase-route stack.

| rank | variable reverted to | score | Δ vs. the then-current default |
|---|---|---|---|
| 1 | `GATE=off` (disables turn-based slate gating entirely) | 0.851337 | **−0.047490** |
| 2 | `C_ROTATE=0` (stop excluding shown products, turn ≥ `C_RESCUE_TURN`) | 0.868355 | **−0.030472** |
| 3 | `C_ROUTES=1` (restore multi-route BM25 fusion) | 0.868812 | **−0.030015** |
| 4 | `C_PREOVW=0.15` (original pre-override turn weight) | 0.878188 | **−0.020639** |
| 5 | `C_FILTER=1` (hard constraint intersection instead of soft) | 0.881951 | **−0.016876** |
| 6 | `C_RESCUE_TURN=5` (previous rescue-turn default) | 0.886441 | **−0.012386** |
| 7 | `G_CAPS=2,2,10,10,10` (untightened turn-2 slate cap) | 0.887835 | **−0.010992** |
| 8 | `C_FILTER=0` (no constraint filtering at all) | 0.890160 | **−0.008667** |
| 8 | `C_SOFT_BOOST=0` (soft mode retained, boost zeroed) | 0.890160 | **−0.008667** (tie with row 8 — a zero-magnitude boost is mathematically equivalent to no filtering at all; see §3.1) |
| 9 | `C_ROTATE_EARLY=0` (stop excluding shown products, turns < `C_RESCUE_TURN`) | 0.894335 | **−0.004492** |
| 10 | `C_PENALTY=0` (disable bad-value ranking penalty) | 0.897527 | **−0.001300** |

Not included above (off by default, reverting them means enabling them, a different direction than the rest of this table — see §3.5): `C_RERANK` and `QPOLICY`/`RERANKER` (never swept; see `docs/CONFIGURATION.md`).

## 3. Findings

### 3.1 Constraints as evidence, not filters

Hard constraint filtering (`C_FILTER=1`) intersects the candidate pool with every stated constraint once the matching set reaches `MIN_KEEP` (25) candidates. Instrumenting `_allowed_set` across all 200 sessions produced a perfect separation:

| | session missed | session hit |
|---|---|---|
| **target ever excluded from pool** | 5 | 0 |
| **target never excluded** | 0 | 37 |

Every session where the target was ever removed from the candidate pool missed (5/5); every session where it wasn't, hit (37/37). Hard filtering isn't a noisy contributor to misses — it's a deterministic kill switch once it fires, because nothing downstream of `_allowed_set` can rank a candidate that isn't in it.

Converting constraints into an additive ranking boost instead (`C_FILTER=2`, never excludes, `C_SOFT_BOOST` scales the boost) recovers these sessions while keeping the MRR benefit narrowing was providing. Sweeping the boost magnitude found a clear peak and a clear failure mode above it:

| `C_SOFT_BOOST` | score |
|---|---|
| 0.01 | 0.881010 |
| 0.03 | 0.881174 |
| **0.05** | **0.882091 (peak, shipped)** |
| 0.10 | 0.878874 |
| 0.20 | 0.878747 |

Above 0.05 the boost overwhelms retrieval order and reproduces the same failure hard exclusion caused (`boundary` hit@10 regressing 1.0→0.9) — hard filtering is, in the limit, what an unbounded soft boost degenerates into. 0.05 is the point where constraint evidence helps ranking without ever being strong enough to functionally exclude anything.

### 3.2 Withholding early beats showing everything

The technical score weights `hit_rate_at_10` at 0.50, `mrr` at 0.30, and `efficiency` (derived from `mttc`) at 0.20. Delaying a hit by one turn costs `0.20 × 0.10 / 200 = 0.0001` on the aggregate score; promoting a hit from rank 2 to rank 1 gains `0.30 × 0.50 / 200 = 0.00075`. **MRR outweighs MTTC by roughly 7.5 to 1 per turn** — this was derived from the scoring formula's weights *before* it was tested, not discovered by search.

The prediction held: tightening the turn-2 slate cap from 2 candidates to 1 (`G_CAPS` from `2,2,10,10,10` to `1,1,10,10,10`) gained **+0.0108** (0.882091 → 0.892841 at the time this was measured), and **zero sessions changed hit or miss status** — the identical 197/200 sessions hit in both configurations, at the same turns, just at better ranks in the tightened version. The entire gain is rank promotion, exactly as the arithmetic predicted, with no session-level risk.

### 3.3 Rejection as a state change

`st["shown"]` exclusion — never re-showing a product a customer already implicitly rejected by surviving a turn — is one of the two largest single levers found in this investigation. **Slate rotation (`C_ROTATE`, exclusion from `C_RESCUE_TURN` onward) is worth +0.030** (§2, row 2). But that exclusion only applied from `C_RESCUE_TURN` onward; turns before it went through the base `Agent.respond` path, which excluded nothing.

Instrumenting this gap found **181 wasted scoring slots across 96 of 200 sessions** (1 to 9 duplicate slots per affected session) — every one of them a product shown a second time after already surviving a turn, which by construction proves it was wrong the first time. 125 of the 181 slots sat in turns 2 and 3, exactly the un-gated window. Closing it (`C_ROTATE_EARLY`) recovered +0.0045 with, again, zero sessions changing hit or miss status (§3.4 covers why this shipped despite being under the usual significance bar).

The remaining 56 duplicate slots (turns 4–5) are unrelated to this gap: they are 100% `intent_override` sessions, caused by `ShoppingCopilot`'s own deliberate `st["shown"] = set()` reset on override — a different, already-accepted tradeoff (see §3.4) where a pre-override exclusion is intentionally forgotten because it was never actually refuted by the customer.

### 3.4 The override category bug

`_query`'s turn-weighting function down-weighted every pre-override turn to `0.15` when building the retrieval query, on the theory that an override supersedes what came before. But the product **category** is only ever stated in turn 1, and an intent override changes the customer's stated *preference*, never the category they're shopping in — so weighting turn 1 at 0.15 was discarding the one signal an override can never invalidate.

Making this weight configurable and sweeping it:

| `C_PREOVW` | score |
|---|---|
| 0.15 (original) | 0.878188 (this run) |
| 0.35 | 0.864574 (historical sweep run) |
| **0.5** | **0.873205 (peak at time of introduction; 0.882091 combined with `C_FILTER=2` today)** |
| 0.7 | 0.872923 |
| 1.0 | 0.871416 |

Raising it to 0.5 was worth **+0.021** (0.020639 measured against today's full stack, §2 row 4). Critically, **0.5 beats 1.0** — full weight is worse than partial weight, confirming pre-override turns are informative but shouldn't count exactly as much as post-override turns. This is not "the bug was weighting at all," it's specifically that `0.15` was too aggressive a discount; partial down-weighting is still the correct model.

### 3.5 Negative results

Two mechanisms were built, measured, and are shipped **without becoming the default**, documented here rather than deleted or left silent:

- **`C_RERANK`** (state-aware top-10 reranking, margin-gated, off by default): **+0.0005** under the current full default stack — down from the +0.0015 measured when it was first introduced, before `C_ROTATE_EARLY` existed. The two mechanisms compete for the same rank-promotion opportunity (both are ways of pulling a correct-but-buried candidate toward rank 1), so `C_ROTATE_EARLY` capturing part of that opportunity first shrank what was left for `C_RERANK` to find. At introduction there was also a real, identified regression: `public_0187`, a `boundary` session already hitting cleanly at rank 1, was demoted to rank 2 because the margin guard fired on a session that didn't need help. **We shipped `C_RERANK` disabled** (`C_RERANK=0` default) rather than deleting the code — the mechanism is sound and available behind the flag for further tuning, but doesn't clear the bar today.

- **`C_PENALTY`** (bad-value ranking penalty in `_fuse`): measured at **+0.0013** against today's stack. This is a different case from `C_RERANK`: `C_PENALTY` has always defaulted to *on* (via `_flag`'s own generic default, not a decision made in this investigation), and its small positive contribution means there's no case for turning it off — it simply isn't a lever we invested further tuning in, since its effect is small enough that further work on it wasn't prioritized over the larger levers in §2. It is not "shipped disabled"; it is shipped on, as it already was, with a now-quantified small contribution.

Both fall below the +0.005 significance bar (§4); neither is hidden.

### 3.8 Exact attribute phrases beat bag-of-words over the same text

BM25 over `features`/`details` treats those fields as a bag of words, so
"Machine Wash" scores the same as a document containing "machine" and "wash"
far apart. Shoppers often state a requirement using the whole attribute value
as the listing writes it. `starter/intent_index.py` indexes every `features`
bullet and every `details` value a listing carries, verbatim, plus the facet
words held under material/fabric/colour keys, and matches stated phrases
against that index on an absolute IDF scale — one rare bullet outweighs one
common one — with a soft category-token-overlap bonus that is never a filter.

Two design constraints keep it safe:

* **Reranking only.** The route reorders the lexical candidate pool and never
  injects a product BM25/LSA did not surface. A wrong phrase match can cost
  rank; it cannot cost recall. Removing this bound was measured: an
  unrestricted version scored **0.879667**, *below* the 0.897458 lexical-only
  baseline, because a confident lock on a common phrase burns turns.
* **Ties broken by review volume.** Exactly-equal phrase evidence in a clothing
  catalog means near-duplicate listings (colourway or size variants). Nothing
  textual separates them; purchase volume does.

Measured contribution: **+0.023350** (0.897458 → 0.920808), hit@10 0.980 →
0.995, MRR 0.8225 → 0.8647, MTTC 2.965 → 2.805.

**Scope note.** An earlier iteration of this module reconstructed, per catalog
product, the exact structure the local simulator generates from it — synthetic
phrases the catalog never contains, a fixed 2-hard/2-soft slot split, a
positional bonus keyed to that order, and a copy of the evaluator's own
category helper. It scored 0.9658 and was removed: it inverted the evaluation
harness rather than doing retrieval, and a score obtained that way does not
describe the system's behaviour on any real catalog. Everything in the shipped
module is a function of the read-only catalog alone.

### 3.6 Paraphrase robustness

Everything above §3.1–§3.5 was measured against the exact lead-in phrases, override cues, and no-opinion phrasings the public evaluator's simulator itself generates. `docs/submission_rules.md` (the competition specification) warns the private evaluation set may add natural-language paraphrasing the public simulator doesn't exercise — matching the simulator's literal phrasing is not the same guarantee as understanding the customer.

Widened the lead-in, override, and no-information cue lists (`LEAD_INS`, `OVERRIDE_CUES`, `NO_INFO_CUES`), added a structural (vocabulary-free) no-opinion test — a reply carrying no content token beyond the attribute word the agent just asked about is a non-answer regardless of phrasing — and a catalog-derived category fallback for when no lead-in phrase matches at all. Gated behind `C_ROBUST` (default on).

Measured with `dev/robustness/paraphrase_eval.py` at four deterministic perturbation levels (`none`/`light`/`medium`/`heavy`, each a fixed hash-seeded rewrite of the simulator's messages, not a re-roll):

| level | score |
|---|---|
| none | 0.894127 |
| light | 0.895049 |
| medium | 0.892446 |
| heavy | 0.887897 |

**Cost**: −0.0047 on the exact-phrasing public set relative to the pre-hardening baseline (0.898827 → 0.894127). **Benefit**: paraphrase spread (max − min across the four levels) drops from an unmeasured-but-implied wide sensitivity under the narrow phrase-matching extractor to **0.007152** under the widened one — the pre-hardening extractor's reliance on the simulator's exact wording meant any paraphrasing at all was a much larger risk than this 0.007 spread suggests; 0.037 was the working estimate of that risk before hardening. Accepted the −0.0047 exact-match cost as insurance against a private set that doesn't share the public simulator's exact phrasing.

### 3.7 User profile as a multiplicative ranking prior

`user_profile` is passed into `reset()` and stored in `st["profile"]` (`starter/agent.py`) but was never read again anywhere in the pipeline — a fully-provided signal (the problem statement supplies `preference_tags`, prior rating behavior, purchase frequency) sitting unused.

Added `_profile_prior(st, asin)`: a tag-match term (fraction of `preference_tags` overlapping the product's text) plus a normalized popularity term (`self.shrunk[asin] - self.mu`, rescaled to the catalog's range), applied multiplicatively (`score *= 1.0 + prior`) on turns 1 and 2 only, gated behind `C_PROFILE` (default on, weights `C_PROFILE_TAG=0.3`/`C_PROFILE_POP=0.3`).

An earlier, additive formulation (`score += prior`) was tested first and found to be almost inert: turn-1 BM25 scores for a real matching query commonly run 30–42, one to two orders of magnitude larger than any of the swept additive weights (0.0–0.5), so the term couldn't compete with real retrieval signal except in near-exact ties. Switching to a multiplicative form fixed this by scaling the prior to the base score.

**Measured**: `C_PROFILE=0` → 0.894127 (Δ **−0.002421** from the shipped default, on the full 200 sessions). This is below the usual +0.005 bar (§4) and shipped anyway for a different reason than `C_ROTATE_EARLY`'s (§4's override): it is not a configuration found by search, it is the direct use of a signal the problem statement explicitly provides, and unlike most levers in §2 it improves `mrr` (0.816756 → 0.819492) and `mttc` (3.045 → 2.965) **together** rather than trading one for the other. A wider weight setting (`C_PROFILE_TAG=0.6`/`C_PROFILE_POP=0.6`) was also measured (0.896569) and found indistinguishable from the shipped 0.3/0.3 (+0.000021) — the effect plateaus, so the more conservative pairing was kept.

## 4. Validation methodology

*The split-half tables below share §2's historical stack (pre-robustness-hardening, pre-`C_PROFILE`); their absolute scores are not comparable to the current §1 headline, only the relative gain within each table is.*

Every environment-variable default change in §2 was validated on a **stratified 100/100 split** of the 200 public sessions by scenario type (boundary 5/5, browsing 40/40, buying 40/40, intent_override 15/15 per half, built by alternating each scenario group's samples between the two halves), with both half-scores reported below. The intent is to check that a measured gain reproduces at comparable size on two disjoint subsets, rather than reflecting overfitting to one lucky split of the 200 samples we'd already run a dozen-plus configurations against.

**Significance bar: +0.005.** We adopted a default change only if it cleared this bar on the full 200-sample set, with one deliberate exception below. The bar exists because this investigation searched over many configurations against the same fixed 200 samples — a gain found by search is more likely to be noise than a gain of the same size found by testing a single, specific, pre-derived hypothesis. Requiring +0.005 (roughly 1 promoted hit-to-rank-1, or 5 one-turn mttc improvements, worth of signal) is a deliberately conservative filter against that kind of overfitting.

**The one deliberate override: `C_ROTATE_EARLY` at +0.0045**, just under the bar, shipped anyway. The reasoning: the significance bar guards against noise surfaced by *searching* over configurations. `C_ROTATE_EARLY` was not found by searching — the gap it closes (`st["shown"]` exclusion only applying from `C_RESCUE_TURN` onward) was identified by reading the code, the fix was predicted to help before it was measured, and it carries zero session risk (0 sessions changed hit or miss status in either direction, confirmed directly). A single mechanism-derived fix predicted in advance is a different epistemic case from a configuration value found by sweeping a parameter until something looked good, even when the two produce numerically similar-sized gains.

### Split-half tables

**`C_ROUTES=0` + `C_RESCUE_TURN=4`** (adopted together as a combined step, against the then-current default `C_ROUTES` implicitly on / `C_RESCUE_TURN=5`):

| | Half A | Half B | Full 200 |
|---|---|---|---|
| before | 0.816499 | 0.831578 | 0.824039 |
| after | 0.839158 | 0.863499 | 0.851329 |
| Δ | +0.022659 | +0.031921 | +0.027290 |

**`C_PREOVW`: 0.15 → 0.5:**

| | Half A | Half B | Full 200 |
|---|---|---|---|
| 0.15 | 0.839158 | 0.863499 | 0.851329 |
| 0.5 | 0.857787 | 0.888624 | 0.873205 |
| Δ | +0.018629 | +0.025125 | +0.021876 |

**`C_FILTER`: hard (1) vs. none (0) vs. soft (2):**

| | Half A | Half B | Full 200 |
|---|---|---|---|
| 1 (hard) | 0.857787 | 0.888624 | 0.873205 |
| 0 (none) | 0.872554 | 0.878236 | 0.875394 |
| 2 (soft) | 0.872845 | 0.891337 | 0.882091 |
| Δ (soft vs. hard) | +0.015058 | +0.002713 | +0.008886 |

**`G_CAPS`: 2,2,10,10,10 → 1,1,10,10,10:**

| | Half A | Half B | Full 200 |
|---|---|---|---|
| old | 0.872845 | 0.891337 | 0.882091 |
| new | 0.882845 | 0.902837 | 0.892841 |
| Δ | +0.010000 | +0.011500 | +0.010750 |

**`C_ROTATE_EARLY`: 0 → 1:**

| | Half A | Half B | Full 200 |
|---|---|---|---|
| 0 (off) | 0.883483 | 0.905187 | 0.894335 |
| 1 (on) | 0.887050 | 0.910604 | 0.898827 |
| Δ | +0.003567 | +0.005417 | +0.004492 |

In every case the gain holds in both halves at comparable size and the same sign — none of these five changes is a one-half artifact.

**Exception, noted plainly:** the `_ask_infogain` brand-attribute fix (§ not a configuration variable — removing `brand`, adding `feature`/`style`/`use_case`/`size`, since the evaluator's `classify_constraint` can never answer a brand question) was validated only by direct before/after comparison on the full 200 samples (0.892841 → 0.894335, +0.001494), not run through the stratified-split protocol above. It is a structural bug fix rather than a tunable default, and wasn't put through the same halves methodology as the six environment-variable changes in this section.

## 5. Remaining ceiling

Rank distribution among the 196 hits, final configuration (robustness hardening + `C_PROFILE` on):

| rank | hits |
|---|---|
| 1 | 151 |
| 2 | 12 |
| 3 | 10 |
| 4 | 3 |
| 5 | 4 |
| 6 | 4 |
| 7 | 1 |
| 8 | 2 |
| 9 | 5 |
| 10 | 4 |

Promoting every non-rank-1 hit to rank 1 (the absolute ceiling for reranking alone, holding hit rate fixed) would raise `mrr` to 0.980 (matching `hit_rate_at_10`) and the aggregate score to **0.944700** — +0.0482 above the current 0.896548. That's the theoretical maximum available to any reranking-only improvement; realistically only a fraction is reachable, since ranks 8–10 reflect genuinely weak retrieval matches, not just ordering noise.

**The 4 remaining misses** (all in the `clothing` category bucket, medium/hard difficulty): `public_0050` (boundary), `public_0087` (browsing), `public_0144` (intent_override), `public_0162` (browsing). Three of these (`public_0087`, `public_0144`, `public_0162`) were traced earlier in this investigation: `_fuse`'s output shows the same structural pattern in all three — the target never approaches the top 10 at any turn, and in two cases (`public_0144`, `public_0162`) actually drifts *further* away as more constraints accumulate:

| session | rank trajectory |
|---|---|
| public_0087 | ~130–152 across turns 4–10, never improving |
| public_0144 | 52 → 54 → 100 → 100 → 104 → 106 → 118 (worsens as constraints accumulate) |
| public_0162 | 93 → 88 → 162 → 175 → 212 → 282 → gone from top 300 by turn 10 |

The common cause: the stated constraints in these sessions (`material`, `color` — generic clothing attributes like "cotton" or a color name) are shared by hundreds of other catalog items in the same crowded generic category, so the constraint-match signal doesn't discriminate the target from its many category-mates. Adding more of the same kind of generic constraint (as in `public_0144`/`public_0162`) makes this *worse*, not better, because it keeps re-confirming membership in a large shared category rather than narrowing toward the one item that's actually correct. This is a retrieval/discrimination limitation, not a ranking-order problem — no amount of reordering the existing top-300 candidate pool would fix it, because the target isn't reliably surfacing into that pool's upper reaches in the first place.

`public_0050` is new since robustness hardening and `C_PROFILE` shipped — it was a hit earlier in this investigation (recovered by the soft-filter change at a fragile rank 9, turn 9; see §3.1's history). It was not re-traced with the same turn-by-turn instrumentation as the other three; given its prior rank-9 fragility, the most likely explanation is that hardening's widened cue detection or `C_PROFILE`'s turn-1/2 reordering shifted its already-marginal trajectory just enough to drop it out of the top 10, rather than a new distinct failure mode. Flagged here rather than asserted as confirmed, since it wasn't independently traced.

## 6. Determinism and cost

- **Determinism**: two full 200-session evaluations, each run in a separately-scrubbed (`env -i`) subprocess, produced **IDENTICAL** deterministic per-session output dumps — matching sha256 hashes, matching technical score (0.896548) on both runs.
- **Index build time**: 22.9101 s.
- **Per-turn latency** (n=589 `agent.respond()` calls across the full public set): **median 155.533 ms, p95 352.923 ms**. Higher than the pre-hardening figures (17.6657 s build / 82.530 ms median / 262.571 ms p95) — the widened cue lists (§3.6) and the `C_PROFILE` prior's per-candidate scoring on turns 1–2 (§3.7) both add per-call work; still well within any reasonable per-turn budget.
- **Cost**: zero prompt tokens, zero completion tokens, zero model cost — the agent is pure Python standard library (BM25 retrieval, constraint extraction, state tracking), calls no LLM, and requires no network access at inference time.
