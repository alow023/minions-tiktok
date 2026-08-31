"""Count extraction-path failures under each perturbation level."""
import json, sys, re
import evaluator.local_evaluator as ev
from starter import agent as ag
from dev.robustness import paraphrase_eval as pe

STATS = {}

def instrument(A):
    class Probe(A):
        def _extract_category(self, msg):
            m = ag.LEAD_RE.search(msg)
            STATS["lead_ok" if m else "lead_fail"] = STATS.get("lead_ok" if m else "lead_fail", 0) + 1
            return super()._extract_category(msg)
        def _is_override(self, msg):
            r = super()._is_override(msg)
            STATS["ovr_true" if r else "ovr_false"] = STATS.get("ovr_true" if r else "ovr_false", 0) + 1
            return r
        def _is_noninformative(self, msg):
            r = super()._is_noninformative(msg)
            if re.search(r"preference|judgment|judgement|whatever|nope|nothing|surprise|easy on|fussed|up to you|doesn't matter|call\b", msg, re.I):
                STATS["noinfo_expected"] = STATS.get("noinfo_expected", 0) + 1
                if not r: STATS["noinfo_MISSED"] = STATS.get("noinfo_MISSED", 0) + 1
            return r
    return Probe

Probe = instrument(ag.Agent)

for lv in sys.argv[1:] or ["none","light","medium","heavy"]:
    STATS.clear()
    a = Probe("data/catalog.jsonl")
    r = pe.run(lv, a)
    # override-turn specific: count intent_override sessions whose override msg detected
    print(lv, json.dumps(STATS), "score", r["recommended_technical_score"], flush=True)
