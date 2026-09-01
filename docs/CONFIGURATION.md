# Configuration

`starter/agent.py`'s `ShoppingCopilot` (aliased to `Agent`, the class the
evaluator and CI actually import) is tuned entirely through environment
variables, each with a hardcoded default chosen from measurement on the
200-sample public set (`data/public_set.jsonl`).

**With no environment variables set, the current defaults reproduce
`recommended_technical_score = 0.920808`** (`hit_rate_at_10 = 0.995`,
`mrr = 0.864692`, `mttc = 2.805`, `efficiency = 0.8195`) — verified with
`env -i python3 -m evaluator.local_evaluator` in a fully scrubbed environment,
with `scikit-learn` installed. Without it the semantic route degrades out and
the agent still runs on the standard library alone.

Measurements labelled *historical* below were taken under the pre-phrase-route
stack (`C_FP=0`, score `0.897458`) and are kept because they explain why each
setting is where it is; they are not deltas against the current default.

The measurements below that predate the robustness defaults are retained as
historical ablations; their numeric deltas are not comparisons against the
current robust baseline. Each was obtained by changing *only*
that one variable away from its default, with every other variable left at
the then-current default stack fixed.

## Retrieval and pipeline structure

### `C_ROBUST` — default on

Enables paraphrase-tolerant opening, override, and no-opinion detection, plus
a catalog-derived fallback for category extraction. The current baseline above
was measured with this setting enabled.

### `C_VOCAB` — default `"0"` (off)

When enabled, adds catalog-mined detail-key vocabularies and expands the
existing style/fit/occasion vocabularies. It remains opt-in because the
robustness baseline does not require it.

### `C_ROUTES` — default `"0"` (off)
Controls whether `_fuse`'s multi-route BM25 retrieval (query route + stated-constraint route + last-turn route) is kept, or collapsed to just the first route. `"0"` collapses to a single route; any other value keeps all routes.
**Measured**: `C_ROUTES=1` → 0.868812 (Δ **−0.030015**). The extra routes hurt more than they help on this dataset — this was the first lever found in the investigation and remains one of the largest.

### `C_RESCUE_TURN` — default `"4"`
The turn number at which `ShoppingCopilot.respond` switches from the simple base-`Agent` BM25 path to its own constraint-aware multi-route pipeline (`_allowed_set`/`_fuse`/gating). Turns before this always use the base path.
**Measured**: `C_RESCUE_TURN=5` → 0.886441 (Δ **−0.012386**). Swept 3/4/5/6 during development; 4 combined with `C_ROUTES=0` was the best pairing found (see git history on this file for the fuller 3–6 sweep table under earlier default stacks).

### `GATE` — default `"margin"`
Selects the per-turn slate-size gating strategy in `_gate_count`. `"margin"` uses the score-band + `G_CAPS` logic described below; `"off"` always returns a full 10-item slate, skipping gating entirely.
**Measured**: `GATE=off` → 0.851337 (Δ **−0.047490**). The single largest lever in this table — early-turn slate withholding is doing substantial work, primarily through MRR (dropping from 0.822 to 0.635 with gating off).

### `G_CAPS` — default `"1,1,10,10,10"`
Comma-separated per-turn slate-size caps consumed by `_gate_count` (only takes effect when `GATE != "off"`). Turn *i* is capped at `caps[min(i, len(caps)) - 1]`, so the last value covers every turn beyond the list; the actual returned count is `max(floor, min(cap, n))` where `n` is the count of same-band-scoring candidates and `floor` preserves the original hardcoded floors (1 for turns ≤2, 3 for turns ≤4, 10 beyond) capped by `cap` itself.
**Measured**: `G_CAPS=2,2,10,10,10` (the pre-tightening value) → 0.887835 (Δ **−0.010992**). The gain comes from promoting turn-2 hits to rank 1 rather than rank 2; MRR outweighs MTTC by roughly 7.5:1 in the scoring formula's weights, which is why withholding an extra candidate on turn 2 pays for itself. Zero sessions changed hit/miss status when this was tightened — the entire gain is rank promotion within already-converting sessions.

### `C_ROTATE_EARLY` — default `"1"` (on)
Excludes products already in `st["shown"]` from the ranking *before* `C_RESCUE_TURN` (the base-`Agent` path), closing a gap where only turns ≥ `C_RESCUE_TURN` excluded already-shown products. `"0"` disables exclusion (the original behavior); `"1"` excludes; `"2"` additionally resets `st["shown"]` when an override is detected in this early path — redundant under `ShoppingCopilot` (see below) but relevant if the base `Agent` is ever used standalone.
**Measured**: `C_ROTATE_EARLY=0` → 0.894335 (Δ **−0.004492**). Fixed 181 wasted scoring slots across 96 of 200 sessions (0 to 9 duplicate slots per affected session); 125 of those slots were in turns 2–3, the exact window this closes. Zero sessions changed hit/miss status in either direction — this shipped despite the gain falling just under the usual +0.005 bar, because that bar guards against noise from searching over configurations, and this is a single mechanism-derived fix predicted in advance with zero session risk, not a search result.

### `C_ROTATE` — default on (via `_flag`'s own `"1"` default; no dedicated default line)
The `C_RESCUE_TURN`-and-later analogue of `C_ROTATE_EARLY`: excludes `st["shown"]` from the slate once `ShoppingCopilot`'s own pipeline is active. Set to `"0"` to disable.
**Measured**: `C_ROTATE=0` → 0.868355 (Δ **−0.030472**). One of the largest levers — most of the benefit of shown-exclusion lives in this later-turn path, with `C_ROTATE_EARLY` closing the smaller remaining gap in turns 1–3.

## Constraint filtering

### `C_FILTER` — default `"2"` (soft)
Controls how stated constraints (material/color/style/fit/occasion/budget) affect the candidate pool in `_allowed_set`/`_fuse`. `"1"`: hard intersection — once a constraint bucket's matching set reaches `MIN_KEEP` (25) candidates, the pool is hard-intersected with it. `"0"`: no filtering at all — constraints are ignored for pool membership. `"2"`: soft — the pool is never narrowed; instead every matched constraint contributes an additive ranking boost (see `C_SOFT_BOOST`).
**Measured**: `C_FILTER=1` → 0.881951 (Δ **−0.016876**); `C_FILTER=0` → 0.890160 (Δ **−0.008667**). Instrumentation found a perfect separation under hard filtering: every session where the target was ever excluded from the pool (5/5) missed, and every session where it wasn't (37/37) hit — hard filtering is a deterministic kill switch, not a noisy contributor. Soft mode recovers those sessions while keeping (and exceeding) the MRR benefit narrowing was providing: +0.008 MRR over hard filtering, +0.011 over no filtering at all (measured at the time soft mode was introduced). Only ~42 of 200 sessions ever reach a turn where this logic is active at all (turn ≥ `C_RESCUE_TURN`) — bounded blast radius, not a dominant factor in the overall score.

### `C_SOFT_BOOST` — default `"0.05"`
Only active when `C_FILTER=2`. Magnitude of the additive ranking boost per matched constraint bucket in soft mode, normalized by the fraction of stated buckets a candidate matches (full match → full boost, zero match → zero boost, never negative).
**Measured**: `C_SOFT_BOOST=0` (with `C_FILTER=2`) → 0.890160 (Δ **−0.008667**, identical to `C_FILTER=0` — a zero-magnitude boost reduces soft mode to plain unfiltered retrieval, as expected). Swept 0.01/0.03/0.05/0.1/0.2/0.4 when soft mode was introduced; 0.05 was the peak — above it the boost overwhelms retrieval order and starts reproducing hard filtering's own failure mode (`boundary` hit@10 regressing) on this dataset, i.e. hard filtering is effectively the limiting case of an unbounded boost.

### `C_PREOVW` — default `"0.5"`
Weight applied to turns before an intent override when building the BM25 query in `_query` (`w = C_PREOVW if turn_no < override_at else 1.0`). The category is only ever stated in turn 1, and an override changes the customer's preference, not the product category — so fully discarding pre-override turns (the original hardcoded `0.15`) throws away that signal.
**Measured**: `C_PREOVW=0.15` (the original hardcoded value) → 0.878188 (Δ **−0.020639**). Swept 0.15/0.35/0.5/0.7/1.0 when this was introduced; 0.5 was the peak — full weight (1.0) still beats 0.15 but is worse than 0.5, since pre-override turns are informative but shouldn't count exactly as much as post-override ones. This variable is inert outside `intent_override` sessions by construction (`override_at` is only ever set on override).

### `C_PENALTY` — default on (via `_flag`'s own `"1"` default; no dedicated default line)
In `_fuse`, multiplies a candidate's score by `0.93 ** min(flags, 3)` for each attribute value it shares with entries in `st["bad_values"]` (values seen ≥2 times across rejected slates and not currently stated). Set to `"0"` to disable.
**Measured**: `C_PENALTY=0` → 0.897527 (Δ **−0.001300**). Small effect under the current stack; a much earlier measurement (under a different default stack, before several other changes) found this completely inert (bit-identical with and without it) because `bad_values` rarely reached the required threshold before a session resolved. It now has a small but real effect.

## Attribute-phrase reranking route

Implemented in `starter/intent_index.py`, fused in `Agent._fp_blend`. See
`docs/ABLATION.md` §3.8 for the mechanism and the measurement.

### `C_FP` — default `"1"` (on)
Master switch. `"0"` disables the route entirely and the agent falls back to
the lexical + semantic pipeline.
**Measured**: `C_FP=0` → 0.897458 (Δ **−0.023350**); hit@10 0.995 → 0.980,
MRR 0.8647 → 0.8225, MTTC 2.805 → 2.965.

### `C_FP_W` — default `"12.0"`
Weight on phrase evidence when fused with the max-normalised lexical score.

### `C_FP_REF` — default `"6.0"`
IDF reference scale: roughly the IDF of a phrase carried by 1 in 400 products,
i.e. the point at which a single matched phrase is specific enough to trust.
Evidence is divided by this before blending, so confidence tracks phrase
*rarity* rather than the share of stated phrases matched. Normalising by the
stated total instead (so one common phrase looks as confident as one rare one)
was measured at **0.884550**, below the lexical-only baseline.

### `C_FP_POPW` — default `"1.0"`
Weight on the review-volume prior that orders candidates with exactly equal
phrase evidence, scaled by the route's own confidence so it is inert while
evidence is thin.

### `C_FP_CATBONUS` — default `"1.0"`
Multiplicative bonus proportional to category-token overlap. Never a hard
filter, so a mis-parsed category cannot drop the true product.

### `C_FP_TAU` — default `"0.35"`
Confidence at which the route is considered *locked*: above it, the current
leading product is exempt from shown-exclusion, so a product identified before
a customer's override lands is still there when the override makes it
scorable. The exemption is scoped to the leader only — exempting the whole
shown-set re-shows declined products deeper in later slates.

### `C_FP_SLATE` — default `"gate"`
Slate width once the route has locked. `"gate"` keeps the normal `G_CAPS`
width; an integer widens it. Widening converts sooner but at a worse rank, and
MRR outweighs efficiency roughly 7:1 in the scoring formula.

### `C_FP_PREOVW` — default `"0.5"`
Weight on phrases stated *before* an intent override. An override rewrites a
preference, not the shopping mission, so earlier phrases stay as
weaker-but-real evidence rather than being discarded.

## User profile

### `C_PROFILE` — default `"1"` (on)
Applies `user_profile` (stored at `reset()` time and, before this, never read again) as a multiplicative ranking prior on turns 1 and 2 only: `score *= (1.0 + _profile_prior(...))`. The prior combines a tag-match term (the fraction of `preference_tags` that overlap the product's text blob) weighted by `C_PROFILE_TAG`, and a normalized popularity term (`self.shrunk[asin] - self.mu`, rescaled to the catalog's observed range) weighted by `C_PROFILE_POP`. Set to `"0"` to disable.
**Measured**: `C_PROFILE=0` → 0.894127 (Δ **−0.002421** from the shipped default). This is under the usual +0.005 significance bar and was shipped anyway: it is not a configuration found by search, it is the use of a signal (`user_profile`) the problem statement explicitly provides and that no other part of the pipeline touches, and it improves MRR (0.816756 → 0.819492) and MTTC (3.045 → 2.965) together rather than trading one for the other.

### `C_PROFILE_TAG` — default `"0.3"`
Weight on the tag-match term in `_profile_prior`. Swept 0.0/0.2/0.3 against `C_PROFILE_POP` at 0.0/0.3/0.5 on the 170 non-`intent_override` sessions before the scoring was corrected from additive to multiplicative (see below); on the full 200 sessions post-correction, 0.3 paired with `C_PROFILE_POP=0.3` gave 0.896548, and doubling both to 0.6/0.6 gave 0.896569 — a +0.000021 difference, i.e. the effect plateaus rather than continuing to scale, so the more conservative 0.3/0.3 pairing was kept.

### `C_PROFILE_POP` — default `"0.3"`
Weight on the normalized-popularity term in `_profile_prior`. See `C_PROFILE_TAG` above for the joint sweep. Note: an earlier version of this mechanism applied the prior additively to the raw BM25 score rather than multiplicatively; at that scale (turn-1 BM25 scores commonly run 30–42 for a real matching query) an additive term of 0.0–0.5 was negligible and the sweep showed almost no effect. The current multiplicative form (`score *= 1.0 + prior`) scales with the base score and is what the measurement above reflects.

## Question policy

### `QPOLICY` — default unset
Selects which attribute-asking strategy is used: `"open"` (always ask a generic open question), `"hybrid"` (info-gain pick, falling back to open if entropy is low), `"infogain"` (always use the info-gain pick), `"none"` (never ask, just show results). Read in two places with different fallback defaults: the base `Agent` path (turns before `C_RESCUE_TURN`) defaults to `"open"` when unset; `ShoppingCopilot`'s own path (turns ≥ `C_RESCUE_TURN`) defaults to `"hybrid"` when unset. Setting `QPOLICY` explicitly overrides both paths uniformly to the same value.
**Not swept in this investigation** — no measured effect recorded.

## The `_ask_infogain` attribute set (not an env var)

Not configurable via environment variable, but worth noting alongside the above: `_ask_infogain`'s candidate attribute list is `("material", "color", "budget", "feature", "style", "use_case", "size")`. It used to include `"brand"`, which the evaluator's `classify_constraint` can never answer (it only ever returns `budget`, `material`, `color`, `size`, `style`, `use_case`, or `feature`), so every brand question burned a turn on a guaranteed non-answer. Fixed by removing `brand` and adding the four evaluator-supported categories that weren't previously tried.
**Measured**: fixing this (independent of any env var) moved the baseline 0.892841 → 0.894335 (Δ **+0.001494**), mostly via `intent_override` mrr (0.577024 → 0.633095).
