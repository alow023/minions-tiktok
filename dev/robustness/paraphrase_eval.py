"""Paraphrase-robustness harness. Wraps evaluator message generators without
modifying the evaluator. Usage: python3 -m dev.robustness.paraphrase_eval LEVEL
"""
from __future__ import annotations
import json, re, sys, hashlib
import evaluator.local_evaluator as ev
from starter.agent import Agent

_orig_initial = ev.initial_message
_orig_reply = ev.customer_reply
_orig_behavior = ev.behavior_for


def _pick(options, key):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return options[h % len(options)]


LEADS = ["I need {x}", "Show me {x}", "I'm after {x}", "Hoping to find {x}",
         "Been hunting for {x}", "Can you help me get {x}", "{x} is what I'm after",
         "In the market for {x}", "Trying to track down {x}"]
HEAVY_LEADS = ["{x}, please", "Something along the lines of {x}", "{x} - that's the idea",
               "Thinking {x}", "{x} would be great", "Anything like {x}?"]

REQ = ["It has to be {c}", "Must be {c}", "One thing though: {c}", "Non-negotiable: {c}",
       "Needs to be {c}", "Only if it's {c}"]
HEAVY_REQ = ["{c} - that part matters", "and {c}", "with {c}", "{c} ideally"]

MATTERS = ["What counts here is {m}", "For me that means {m}", "On that front: {m}",
           "Honestly, {m}", "I'd say {m}"]
HEAVY_MATTERS = ["{m}", "{m}, more or less", "probably {m}"]

NOPREF = ["No strong feelings on {a}, your call", "{a} really doesn't matter to me",
          "I'm easy on {a}", "Not fussed about {a}"]
HEAVY_NOPREF = ["{a}? whatever works", "eh, {a} is up to you", "surprise me on {a}"]

NOMORE = ["Nothing else on {a}", "Can't think of anything more about {a}",
          "That's all I've got for {a}"]
HEAVY_NOMORE = ["{a}, nope", "nothing more there"]

NOTRIGHT = ["Those aren't quite it. Ask me about one specific attribute.",
            "Not quite what I had in mind - ask about a single attribute.",
            "Close but no. Pick one attribute and ask me."]
HEAVY_NOTRIGHT = ["nah, try again - ask me something specific.",
                  "not it. one attribute, go."]

OVERRIDE = ["Scratch that - what I really need is: {v}.",
            "Change of plan. Forget what I said; I want: {v}.",
            "On second thought, ignore that. The real requirement is: {v}.",
            "Actually never mind my last message - {v} is what matters."]
HEAVY_OVERRIDE = ["forget the earlier thing. {v}.",
                  "no wait - {v}, that's the one.",
                  "different direction: {v}."]

EXPLORE = ["I need {x}, still weighing options though.",
           "Show me {x} - just browsing for now.",
           "{x} is roughly the idea, nothing settled yet."]
HEAVY_EXPLORE = ["{x}, just poking around.", "browsing {x} for now.", "{x}? window shopping."]

TRACE = {"initial": [], "reply": []}


def make_initial(level):
    def f(sample, category, disclosed):
        base = _orig_initial(sample, category, disclosed)
        key = base
        sc = sample["scenario_type"]
        if sc == "buying" and sample["intent_card"].get("hard_constraints"):
            c = str(sample["intent_card"]["hard_constraints"][0])
            if level == "light":
                out = _pick(LEADS, key).format(x=category) + ". A key requirement is: " + c + "."
            elif level == "medium":
                out = _pick(LEADS, key).format(x=category) + ". " + _pick(REQ, key).format(c=c) + "."
            else:
                out = _pick(HEAVY_LEADS, key).format(x=category) + " " + _pick(HEAVY_REQ, key).format(c=c) + "."
        elif sc == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            if level == "light":
                out = _pick(LEADS, key).format(x=category) + ". " + old
            elif level == "medium":
                out = _pick(LEADS, key).format(x=category) + ". " + _pick(MATTERS, key).format(m=old) + "."
            else:
                out = _pick(HEAVY_LEADS, key).format(x=category) + " " + _pick(HEAVY_MATTERS, key).format(m=old) + "."
        else:
            if level == "light":
                out = _pick(LEADS, key).format(x=category) + ", but I'm still exploring."
            elif level == "medium":
                out = _pick(EXPLORE, key).format(x=category)
            else:
                out = _pick(HEAVY_EXPLORE, key).format(x=category)
        TRACE["initial"].append((category, base, out))
        return out
    return f


def make_reply(level):
    def f(sample, ask_attribute, disclosed, boundary_used):
        msg, bu = _orig_reply(sample, ask_attribute, disclosed, boundary_used)
        key = msg
        out = msg
        m = re.match(r"I don't have a preference for (.+); please use your judgment\.$", msg)
        if m and level != "light":
            a = m.group(1)
            out = _pick(NOPREF if level == "medium" else HEAVY_NOPREF, key).format(a=a) + "."
        else:
            m2 = re.match(r"I don't have an additional preference for (.+)\.$", msg)
            if m2 and level != "light":
                out = _pick(NOMORE if level == "medium" else HEAVY_NOMORE, key).format(a=m2.group(1)) + "."
            elif msg.startswith("Those options are not quite right") and level != "light":
                out = _pick(NOTRIGHT if level == "medium" else HEAVY_NOTRIGHT, key)
            else:
                m3 = re.match(r"For that, what matters is: (.+)\.$", msg)
                if m3:
                    body = m3.group(1)
                    if level != "light":
                        out = _pick(MATTERS if level == "medium" else HEAVY_MATTERS, key).format(m=body) + "."
        TRACE["reply"].append((msg, out))
        return out, bu
    return f


def make_behavior(level):
    def f(scenario, card, rng):
        b = _orig_behavior(scenario, card, rng)
        if scenario == "intent_override":
            v = str(b["override"]["new_value"])
            pool = OVERRIDE if level in ("light", "medium") else HEAVY_OVERRIDE
            b["override"]["message"] = _pick(pool, v).format(v=v)
        return b
    return f


def run(level, agent=None):
    ev.initial_message = _orig_initial
    ev.customer_reply = _orig_reply
    ev.behavior_for = _orig_behavior
    if level != "none":
        ev.initial_message = make_initial(level)
        ev.customer_reply = make_reply(level)
        ev.behavior_for = make_behavior(level)
    samples = ev.load_jsonl("data/public_set.jsonl")
    ids, cats, prods = ev.catalog_index("data/catalog.jsonl")
    res = ev.evaluate(agent or Agent("data/catalog.jsonl"), samples, ids, cats, prods)
    return res


def brief(level, r):
    return {"level": level, "score": r["recommended_technical_score"],
            "hit": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "eff": r["efficiency"],
            "scen": {k: [v["hit_rate_at_10"], round(v["mrr"], 4), v["mttc"]]
                     for k, v in r["scenario_metrics"].items()}}


if __name__ == "__main__":
    levels = sys.argv[1:] or ["none", "light", "medium", "heavy"]
    ag = Agent("data/catalog.jsonl")
    for lv in levels:
        ag.state = {}
        r = run(lv, ag)
        print(json.dumps(brief(lv, r)), flush=True)
        json.dump(r, open(f"dev/robustness/res_{lv}.json", "w"))
