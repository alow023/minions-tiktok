"""DialogController — Person B's piece of the frozen interface contract.

Consumes LanguageFilter output (constraints) and drives per-session state:
exclusion, Rocchio-style soft penalties, and which attribute to ask about
next. See the contract docstring block shared with the team for the exact
method shapes this must satisfy.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from legacy_starter.stub_ranker import rank as stub_rank

LOGGER = logging.getLogger(__name__)

LEGAL_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

# Attributes classify_constraint (in the evaluator) can actually produce.
# 'brand' and 'category' never match a real constraint, so choose_question
# never asks for them; 'other' matches any undisclosed constraint regardless
# of its bucket, so it's the highest-value fallback question.
ASKABLE_ATTRIBUTES = ("budget", "material", "color", "size", "style", "use_case", "feature")

OVERRIDE_RE = re.compile(r"^Actually, ignore my earlier preference\. What I need is:\s*(.+?)\.?\s*$")

# customer_reply() in the evaluator has two distinct "no preference"
# replies that must not be conflated:
#   - "I don't have a preference for {attribute}; please use your
#     judgment." fires only in a boundary session, only once, and reveals
#     nothing about that attribute -- it's not exhausted, the customer
#     just doesn't care. Detected via the substring 'please use your
#     judgment', which only that template contains.
#   - "I don't have an additional preference for {attribute}." fires in
#     any scenario once that attribute's undisclosed constraints run out.
#     Detected via the substring 'additional preference', which only that
#     template contains. That attribute really is exhausted going forward.
BOUNDARY_NO_PREF_RE = re.compile(r"^I don't have a preference for (.+); please use your judgment\.$")
EXHAUSTED_NO_PREF_RE = re.compile(r"^I don't have an additional preference for (.+)\.$")


def _parse_no_preference(message: str) -> tuple[str, str | None] | None:
    """Return ('boundary' | 'exhausted', attribute) for a no-preference
    reply, or None if message is neither template."""
    if "please use your judgment" in message:
        match = BOUNDARY_NO_PREF_RE.match(message)
        return ("boundary", match.group(1) if match else None)
    if "additional preference" in message:
        match = EXHAUSTED_NO_PREF_RE.match(message)
        return ("exhausted", match.group(1) if match else None)
    return None

# The two turn-1 templates that carry a disclosed constraint after the
# category sentence. Order matters in detect_scenario(): BUYING_RE must be
# tried before treating a leading "I'm looking for " as intent_override,
# since both templates share that prefix.
BUYING_RE = re.compile(r"^I'm looking for .+\. A key requirement is: .+\.$")
EXPLORING_RE = re.compile(r"^I'm looking for .+, but I'm still exploring\.$")


def detect_scenario(message: str) -> str:
    """Classify a turn-1 customer message by which of initial_message()'s
    three literal templates (evaluator/local_evaluator.py) produced it.

    Returns 'buying', 'intent_override', or 'exploring'. 'exploring' covers
    both the browsing and boundary scenario types: initial_message() emits
    the exact same template for both ("I'm looking for {category}, but I'm
    still exploring."), so message text alone can't tell them apart -- the
    evaluator already knows scenario_type, this function doesn't need to.
    """
    text = message or ""
    if BUYING_RE.match(text):
        return "buying"
    if EXPLORING_RE.match(text):
        return "exploring"
    if text.startswith("I'm looking for "):
        return "intent_override"
    return "unknown"

# Flat penalty added, per product, when a slate is promoted while still
# inside an unfired intent_override session (see observe()). Not a hard
# exclusion: the product may turn out to be the (superseded) target once
# the override lands, so it's just made less likely to be reshown
# immediately rather than banned outright.
PENALTY_INCREMENT = 0.5
PENALTY_CAP = 0.99

# Rocchio-style negative-feedback weight for attribute values that are
# over-represented in a rejected slate relative to the surviving candidate
# population (see observe()'s _apply_negative_feedback). Kept well below
# 1.0 -- the implicit weight of a positive signal such as a survivor
# actually matching a disclosed constraint -- so accumulated negative
# evidence about a value (e.g. "black was over-represented among misses")
# can discourage but never outweigh or cancel out real positive evidence.
NEGATIVE_FEEDBACK_WEIGHT = 0.25

# choose_question won't ask about an attribute unless its expected-
# information-gain score (see DialogController._balance_score) clears this
# bar. Note that score is Balance(a) * EMPIRICAL_ATTRIBUTE_PRIORS[a] and
# Balance(a) <= 1, so no attribute can ever clear this threshold unless its
# prior alone exceeds it -- 'budget' (0.0016), 'size' (0.0209), 'use_case'
# (0.0053), and 'style' (0.0469) are mathematically incapable of winning
# outright regardless of how perfectly they split the candidates. Only
# 'material' (0.2734) and 'feature' (0.5256) can. That's an intentional
# consequence of weighting by real disclosure frequency, not a bug -- but
# it means those four buckets are asked about only as a side effect of
# 'other' matching whatever the customer discloses, never as the direct
# argmax pick.
BALANCE_THRESHOLD = 0.2

# Empirical frequency of each classify_constraint() bucket across
# hard_constraints + soft_preferences from intent_card(), sampled over 2000
# random catalog products (seed 0). Produced by diagnose_buckets.py --
# re-run that script and update this constant if the catalog or
# intent_card()/classify_constraint() change. Values sum to ~1.0:
#
#   feature     4154  (52.56%)
#   material    2161  (27.34%)
#   color        998  (12.63%)
#   style        371  ( 4.69%)
#   size         165  ( 2.09%)
#   use_case      42  ( 0.53%)
#   budget        13  ( 0.16%)
#
# Used in _balance_score as an "answerability" multiplier: Balance(a)
# measures how evenly attribute a would split the surviving candidates
# (structural, per-turn, computed from candidate_attributes); the prior
# measures how likely the customer actually has a constraint of that type
# at all (historical, global, fixed). Their product approximates expected
# information gain -- a perfect 50/50 split is worthless to ask about if
# customers essentially never have an opinion on it (e.g. 'budget'), and a
# so-so split is still worth asking about if that bucket usually has
# something to disclose (e.g. 'feature').
EMPIRICAL_ATTRIBUTE_PRIORS: dict[str, float] = {
    "feature": 0.5256,
    "material": 0.2734,
    "color": 0.1263,
    "style": 0.0469,
    "size": 0.0209,
    "use_case": 0.0053,
    "budget": 0.0016,
}


def _blank_state() -> dict:
    return {
        "exclude_ids": set(),
        # Penalties are tracked as two separate sources internally, merged
        # only by state() for the frozen contract's single "penalties" key:
        #   - shown_penalties: from slate promotion (observe() step 1). A
        #     product having been displayed and missed is a fact that
        #     stays true even after an intent change.
        #   - feedback_penalties: from Rocchio negative feedback (step 1b).
        #     These are inferences drawn from attribute-value correlations
        #     under the (possibly now-superseded) old intent, so they get
        #     wiped on override while shown_penalties survives.
        "shown_penalties": {},
        "feedback_penalties": {},
        "constraints": [],
        "turn": 0,
        "asked": set(),
        "pending_slate": [],
        "override_fired": False,
        "exhausted": set(),
        "boundary_seen": False,
        "scenario": None,
    }


class DialogController:
    def __init__(self, candidate_attributes: dict[str, dict[str, str]] | None = None) -> None:
        self.sessions: dict[str, dict] = {}
        # parent_asin -> {attribute: value}, one entry per askable attribute
        # a product's catalog text actually mentions. Supplied by whichever
        # component extracts per-product attribute values from the catalog
        # (LanguageFilter, in the full pipeline); choose_question uses it
        # read-only to score candidate splits. Defaults to empty so existing
        # callers that don't pass it still get a valid (if uninformative,
        # always-'other') controller.
        self.candidate_attributes = candidate_attributes or {}
        # Counts how many _balance_score calls found zero coverage for each
        # attribute (Balance had to be estimated rather than measured).
        # Accumulated silently; see log_empty_bucket_summary(). A per-call
        # warning would fire constantly for buckets that are simply rare
        # but working correctly (e.g. 'budget' at a 0.16% prior), drowning
        # the signal for a bucket that's *always* empty because an
        # upstream attribute extractor never populates it.
        self._empty_coverage_counts: Counter[str] = Counter()

    # -- contract methods -------------------------------------------------

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = _blank_state()
        state["user_profile"] = user_profile
        self.sessions[session_id] = state

    def observe(self, session_id: str, message: str, constraints: list[dict]) -> None:
        state = self.sessions.setdefault(session_id, _blank_state())
        state["turn"] += 1

        # detect_scenario() reads the turn-1 message shape; capture it once
        # and keep it, since later turns' messages don't match any of the
        # three initial_message() templates and would otherwise read as
        # 'unknown'.
        if state["scenario"] is None:
            state["scenario"] = detect_scenario(message or "")

        pending = state["pending_slate"]
        unfired_intent_override = state["scenario"] == "intent_override" and not state["override_fired"]

        # 1. promote the previous slate: hard-exclude it, except inside an
        # intent_override session whose override hasn't fired yet, where a
        # pre-override miss proves nothing about the (still hidden) real
        # target -- so it gets a flat penalty instead of a permanent ban.
        if pending:
            if unfired_intent_override:
                for parent_asin in pending:
                    current = state["shown_penalties"].get(parent_asin, 0.0)
                    state["shown_penalties"][parent_asin] = min(current + PENALTY_INCREMENT, PENALTY_CAP)
            else:
                for parent_asin in pending:
                    state["exclude_ids"].add(parent_asin)
                    state["shown_penalties"].pop(parent_asin, None)
                    state["feedback_penalties"].pop(parent_asin, None)

        # 1b. Rocchio-style negative feedback: an attribute value that's
        # over-represented in the just-rejected slate, relative to how
        # often it occurs among the candidates still in play, gets a
        # penalty bump on every surviving candidate that carries it --
        # "the customer kept rejecting black items, so black is
        # discouraged going forward" -- without excluding anyone outright.
        # Placed before override detection: if this message IS the
        # override, step 2 below clears feedback_penalties anyway,
        # correctly discarding feedback computed against the
        # now-superseded intent (shown_penalties survives, see step 2).
        if pending:
            self._apply_negative_feedback(state, pending)

        # 2. detect an override and, if found, drop stale state.
        override_match = OVERRIDE_RE.match(message or "")
        override_fired_this_turn = override_match is not None
        if override_fired_this_turn:
            state["override_fired"] = True
            # We aren't told which single prior constraint the override
            # supersedes, only that one has been. Rather than guess, treat
            # the whole accumulated set as belonging to the old intent and
            # start clean; step 3 below repopulates it from this turn.
            state["constraints"] = []
            # Only feedback_penalties is cleared: those are attribute-value
            # inferences drawn from the now-superseded intent and carry no
            # signal about the new one. shown_penalties survives -- a
            # product having actually been displayed and missed remains
            # true regardless of what the customer wants now.
            state["feedback_penalties"] = {}
            state["asked"] = set()
            state["exhausted"] = set()

        # detect the two no-preference reply shapes (see _parse_no_preference).
        no_preference = _parse_no_preference(message or "")
        if no_preference is not None:
            kind, attribute = no_preference
            if kind == "boundary":
                state["boundary_seen"] = True
            elif kind == "exhausted" and attribute:
                state["exhausted"].add(attribute)

        # 3. merge the new constraints into state.
        for constraint in constraints:
            if constraint not in state["constraints"]:
                state["constraints"].append(constraint)

        state["pending_slate"] = []

    def register_slate(self, session_id: str, shown_ids: list[str]) -> None:
        state = self.sessions.setdefault(session_id, _blank_state())
        state["pending_slate"] = list(shown_ids)

    def _apply_negative_feedback(self, state: dict, rejected_slate: list[str]) -> None:
        """Rocchio-style asymmetric negative feedback: for each attribute,
        compare how often each value occurs in the just-rejected slate
        against how often it occurs among the surviving candidates (every
        product this controller knows about that isn't already
        exclude_ids). A value over-represented in the rejects --
        freq_rejected(v) > freq_surviving(v) -- gets every surviving
        product carrying it penalized by NEGATIVE_FEEDBACK_WEIGHT times
        that frequency difference. Values that are merely as common among
        rejects as everywhere else (diff <= 0) get no penalty: this is
        meant to catch "the customer keeps rejecting black", not to
        punish popularity in general.
        """
        surviving_ids = [pid for pid in self.candidate_attributes if pid not in state["exclude_ids"]]
        rejected_total = len(rejected_slate)
        surviving_total = len(surviving_ids)
        if rejected_total == 0 or surviving_total == 0:
            return

        # Single pass over survivors: attribute -> value -> [surviving ids].
        surviving_index: dict[str, dict[str, list[str]]] = {}
        for pid in surviving_ids:
            attrs = self.candidate_attributes.get(pid) or {}
            for attribute in ASKABLE_ATTRIBUTES:
                value = attrs.get(attribute)
                if value:
                    surviving_index.setdefault(attribute, {}).setdefault(value, []).append(pid)

        # Single pass over the rejected slate: attribute -> value -> count.
        rejected_counts: dict[str, Counter[str]] = {}
        for pid in rejected_slate:
            attrs = self.candidate_attributes.get(pid) or {}
            for attribute in ASKABLE_ATTRIBUTES:
                value = attrs.get(attribute)
                if value:
                    rejected_counts.setdefault(attribute, Counter())[value] += 1

        for attribute, value_counts in rejected_counts.items():
            for value, rejected_count in value_counts.items():
                freq_rejected = rejected_count / rejected_total
                surviving_matches = surviving_index.get(attribute, {}).get(value, [])
                freq_surviving = len(surviving_matches) / surviving_total
                diff = freq_rejected - freq_surviving
                if diff <= 0:
                    continue
                increment = NEGATIVE_FEEDBACK_WEIGHT * diff
                for pid in surviving_matches:
                    current = state["feedback_penalties"].get(pid, 0.0)
                    state["feedback_penalties"][pid] = min(current + increment, PENALTY_CAP)

    def choose_question(self, session_id: str, candidate_ids: list[str]) -> str:
        # For each of the seven attributes classify_constraint can actually
        # emit, score(a) = Balance(a) * EMPIRICAL_ATTRIBUTE_PRIORS[a] is an
        # expected-information-gain estimate:
        #   - Balance(a) = 1 - abs(2 * max_v P(v|C) - 1) measures how evenly
        #     attribute a would split the surviving candidates C. It peaks
        #     at 1 when the most common value covers ~50% of C (asking
        #     roughly halves the field) and falls to 0 when one value is
        #     near-unanimous (asking wouldn't split anything).
        #   - EMPIRICAL_ATTRIBUTE_PRIORS[a] measures how likely a customer
        #     even has a constraint of that type to disclose at all, from
        #     real intent_card()/classify_constraint() frequency.
        # The two are different units -- Balance is about *this turn's*
        # candidate set, the prior is a global historical rate -- but their
        # product is what you'd want to maximize if you actually cared
        # about expected information gain: a perfect split nobody has an
        # opinion on (e.g. 'budget') isn't worth asking, and an unmeasured
        # bucket that's usually informative (e.g. 'feature') can still beat
        # a measured-but-narrow one. See BALANCE_THRESHOLD's comment for
        # the consequence: several buckets can now never win outright no
        # matter how they split, because their prior alone is below it.
        # The argmax wins unless every legal, unasked attribute scores
        # below BALANCE_THRESHOLD, or none remain unasked -- either way
        # 'other' is the fallback, since it matches any undisclosed
        # constraint regardless of bucket (see classify_constraint).
        # Never repeats an attribute except after an override reset, since
        # 'asked' (like 'exhausted') is cleared there and nowhere else.
        try:
            state = self.sessions.get(session_id)
            if state is None:
                return "other"
            asked = state.setdefault("asked", set())
            exhausted = state.setdefault("exhausted", set())

            best_attribute: str | None = None
            best_score = -1.0
            for attribute in ASKABLE_ATTRIBUTES:
                if attribute in asked or attribute in exhausted:
                    continue
                score = self._balance_score(attribute, candidate_ids)
                if score > best_score:
                    best_score = score
                    best_attribute = attribute

            # Tolerance guards against float rounding landing an exact
            # threshold split (e.g. a 90/10 split's Balance*coverage is
            # mathematically 0.2 but computes as 0.19999999999999996).
            if best_attribute is not None and best_score >= BALANCE_THRESHOLD - 1e-9:
                asked.add(best_attribute)
                return best_attribute
            return "other"
        except Exception:
            return "other"

    def _balance_score(self, attribute: str, candidate_ids: list[str]) -> float:
        """score(a) = Balance(a) * EMPIRICAL_ATTRIBUTE_PRIORS[a].

        Balance(a) measures how evenly attribute a splits the surviving
        candidates (1.0 = perfect ~50/50 split, 0.0 = unanimous); the prior
        measures how likely a customer has a constraint of that type at
        all. The product estimates expected information gain: an
        attribute is only worth asking about if it BOTH would split the
        field AND is something customers actually tend to disclose.

        When no candidate in this set has any value for the attribute,
        Balance can't be measured at all -- there's no data to compute
        max_v P(v|C) from. Rather than returning the prior on its own
        (which would conflate "unmeasured" with "measured and found
        useless"), a neutral Balance of 0.5 stands in for "unknown, assume
        middling" and the estimate is flagged (via
        _empty_coverage_counts) for the aggregate summary rather than
        warned about immediately -- see log_empty_bucket_summary().
        """
        total = len(candidate_ids)
        if total == 0:
            return 0.0
        mentioned_values = [
            value
            for cid in candidate_ids
            if (value := self.candidate_attributes.get(cid, {}).get(attribute))
        ]
        prior = EMPIRICAL_ATTRIBUTE_PRIORS.get(attribute, 0.0)
        if not mentioned_values:
            self._empty_coverage_counts[attribute] += 1
            estimated_balance = 0.5
            return estimated_balance * prior
        counts = Counter(mentioned_values)
        max_p = max(counts.values()) / len(mentioned_values)
        balance = 1 - abs(2 * max_p - 1)
        return balance * prior

    def log_empty_bucket_summary(self) -> None:
        """Emit one aggregated warning for every attribute bucket that ever
        had zero candidate coverage during this controller's lifetime,
        with how many times each occurred. Call this once at the end of a
        run (e.g. after a full evaluator pass), not per turn: a per-call
        warning would fire constantly for buckets that are simply rare but
        working correctly (e.g. 'budget' at a 0.16% empirical prior),
        drowning out the signal for a bucket that's genuinely never
        populated because an upstream attribute extractor is broken.
        """
        if not self._empty_coverage_counts:
            return
        summary = ", ".join(
            f"{attribute}={count}" for attribute, count in self._empty_coverage_counts.most_common()
        )
        LOGGER.warning(
            "attribute buckets with zero candidate coverage at least once this run "
            "(occurrences): %s -- if any of these was ALWAYS empty rather than just "
            "occasionally, check the upstream attribute extractor rather than "
            "assuming it's just rare",
            summary,
        )

    def state(self, session_id: str) -> dict:
        # Tolerant like observe()/register_slate() (setdefault, not a
        # raise): an unreset session id gets a fresh blank state rather
        # than an exception, so state() can never be the thing that takes
        # an otherwise-working turn down.
        session = self.sessions.setdefault(session_id, _blank_state())
        # The frozen contract exposes a single "penalties" key; internally
        # it's tracked as two independently-capped sources (see
        # _blank_state's comment). Their sum is re-capped here too, since
        # two values each individually < PENALTY_CAP could otherwise sum
        # past the contract's documented 0.0-to-<1.0 bound.
        shown = session["shown_penalties"]
        feedback = session["feedback_penalties"]
        # NOT `shown.keys() | feedback.keys()`: a set union of strings
        # iterates in an order that depends on PYTHONHASHSEED, which is
        # randomized per-process by default -- the exact same two dicts
        # produce a different key order in every fresh `python3` run. Both
        # `shown` and `feedback` are plain dicts (insertion-ordered,
        # deterministic), so dict.fromkeys() over their concatenated keys
        # preserves that determinism instead of routing through a set.
        merged = {
            parent_asin: min(shown.get(parent_asin, 0.0) + feedback.get(parent_asin, 0.0), PENALTY_CAP)
            for parent_asin in dict.fromkeys((*shown, *feedback))
        }
        return {
            "exclude_ids": set(session["exclude_ids"]),
            "penalties": merged,
            "constraints": list(session["constraints"]),
            "turn": session["turn"],
        }

    # -- turn policy --------------------------------------------------
    #
    # Not part of the frozen five-method contract; this is the glue that
    # guarantees the two hard requirements every turn_response must meet
    # (docs/agent_api_contract.json: recommendations is capped at 100 but
    # the evaluator only scores the first 10, and ask_attribute must never
    # be null). Must be called AFTER observe() for this turn, since it
    # reads state (exclude_ids/penalties) that observe() just updated, and
    # it performs register_slate()/choose_question() itself.

    def take_turn(self, session_id: str, candidate_ids: list[str], top_k: int = 10) -> dict:
        """Rank, top up to exactly top_k, register the slate, and choose
        this turn's ask_attribute. Returns {"recommendations": [...],
        "ask_attribute": str}; the caller still owns "message" and "usage".
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise RuntimeError("reset must be called before take_turn")

        state = self.state(session_id)
        ranked = stub_rank(candidate_ids, state, k=top_k)
        if len(ranked) < top_k:
            ranked = self._top_up(ranked, state["exclude_ids"], top_k)
        self.register_slate(session_id, ranked)

        # Hits are never scored before an intent_override's override fires
        # (evaluate() gates hit detection on override_applied), and the
        # evaluator always fires it at turn 3 or 4. So for turns 1-2 of
        # such a session, a specific attribute question risks extracting
        # disclosure about the soon-to-be-superseded preference for no
        # scoring credit; 'other' matches any undisclosed constraint
        # regardless of bucket, maximizing what gets captured before the
        # override arrives instead of gambling on one Balance-scored bucket.
        if session["scenario"] == "intent_override" and session["turn"] <= 2:
            attribute = "other"
            session.setdefault("asked", set()).add("other")
        else:
            attribute = self.choose_question(session_id, candidate_ids)

        return {
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": parent_asin} for parent_asin in ranked],
        }

    def _top_up(self, ranked: list[str], exclude_ids: set, top_k: int) -> list[str]:
        """Pad a short ranked list from the controller's known candidate
        universe (self.candidate_attributes) so the turn always returns
        top_k recommendations, even if the caller's candidate_ids pool was
        too thin (already fully excluded/ranked) to reach it on its own.
        Best-effort: if the whole known universe has fewer than top_k
        unexcluded products left, returns as many as actually exist.
        """
        selected = list(ranked)
        if len(selected) >= top_k:
            return selected[:top_k]
        seen = set(selected)
        for parent_asin in self.candidate_attributes:
            if len(selected) >= top_k:
                break
            if parent_asin in seen or parent_asin in exclude_ids:
                continue
            selected.append(parent_asin)
            seen.add(parent_asin)
        return selected
