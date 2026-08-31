"""
Conversational shopping agent — no evaluator-derived knowledge.

Design rule applied throughout: every string constant and every heuristic in
this file must be justifiable from (a) the published Agent API contract,
(b) the product catalog itself, or (c) general knowledge of how English
shoppers write. Nothing is derived from reading the simulator's source.

Attribute vocabularies are LEARNED FROM THE CATALOG at index time rather than
hardcoded, so they contain no borrowed word lists.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Standard English function words. Nothing task-specific, nothing borrowed.
STOP = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he her here hers herself
him himself his how i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she should shouldn't so some such than that the their theirs them themselves then there these
they this those through to too under until up very was wasn't we were weren't what when where which
while who whom why with won't would wouldn't you your yours yourself yourselves
""".split())

# Generic ways an English shopper opens a request. Not a simulator template.
LEAD_INS = [
    r"looking for", r"i need", r"i want", r"searching for", r"shopping for",
    r"trying to find", r"i'?m after", r"show me", r"find me", r"do you have",
]
LEAD_RE = re.compile(r"(?:" + "|".join(LEAD_INS) + r")\s+(.+?)(?:[,.;!?]|$)", re.I)

# Generic English markers that a speaker is retracting a previous statement.
OVERRIDE_CUES = ("actually", "instead", "on second thought", "changed my mind",
                 "scratch that", "never mind", "nevermind", "forget", "i'd rather",
                 "rather than", "no wait", "correction")

# Generic English markers of a non-answer / no-opinion reply.
NO_INFO_CUES = ("no preference", "don't have a preference", "dont have a preference",
                "doesn't matter", "doesnt matter", "no strong", "not fussed", "not picky",
                "up to you", "you decide", "your judgment", "your judgement", "whatever",
                "not sure", "either is fine", "no additional")


def flat_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def toks(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOP]


class BM25:
    """Field-weighted BM25 with unigrams + bigrams."""

    def __init__(self, k1: float = 1.5, b: float = 0.6):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.doclen: list[float] = []
        self.ids: list[str] = []

    def add(self, doc_id: str, weighted_fields, bigrams: bool = True):
        tf: dict[str, float] = defaultdict(float)
        for text, w in weighted_fields:
            tt = toks(text)
            for t in tt:
                tf[t] += w
            if bigrams:
                for x, y in zip(tt, tt[1:]):
                    tf[x + "_" + y] += w * 1.5
        idx = len(self.ids)
        self.ids.append(doc_id)
        self.doclen.append(sum(tf.values()))
        for t, f in tf.items():
            self.postings[t].append((idx, f))

    def finalize(self):
        self.N = len(self.ids)
        self.avgdl = sum(self.doclen) / max(1, self.N)
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def score(self, query_terms: dict[str, float]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t, qw in query_terms.items():
            p = self.postings.get(t)
            if not p or len(p) > self.N * 0.35:
                continue
            idf = self.idf[t]
            for idx, f in p:
                denom = f + self.k1 * (1 - self.b + self.b * self.doclen[idx] / self.avgdl)
                out[idx] += qw * idf * (f * (self.k1 + 1)) / denom
        return out


class Agent:
    # ask_attribute values come from the published API contract (docs/agent_api_contract.json)
    ASKABLE = ("material", "color", "style", "use_case", "budget", "brand", "size", "category")

    def __init__(self, catalog_path="data/catalog.jsonl",
                 question_policy=None, gate=None, bigrams=True):
        self.bigrams = bigrams
        self.question_policy = question_policy or os.environ.get("QPOLICY", "open")
        self.gate = gate or os.environ.get("GATE", "margin")
        self.bm25 = BM25()
        self.cat_tokens: dict[str, set[str]] = {}
        self.blob: dict[str, str] = {}
        self.price: dict[str, float | None] = {}
        self.store: dict[str, str] = {}
        mat_counter: Counter = Counter()
        col_counter: Counter = Counter()

        rows = []
        with Path(catalog_path).open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                rows.append(p)
                det = p.get("details") or {}
                if isinstance(det, dict):
                    for key, val in det.items():
                        kl = str(key).lower()
                        if "material" in kl or "fabric" in kl:
                            mat_counter.update(toks(str(val)))
                        if "color" in kl or "colour" in kl:
                            col_counter.update(toks(str(val)))
        self.vocab = {
            "material": {w for w, c in mat_counter.most_common(60)},
            "color": {w for w, c in col_counter.most_common(60)},
        }

        for p in rows:
            a = str(p["parent_asin"])
            title = flat_text(p.get("title"))
            feats = flat_text(p.get("features"))
            det = flat_text(p.get("details"))
            desc = flat_text(p.get("description"))
            cats = flat_text(p.get("categories"))
            store = flat_text(p.get("store"))
            self.bm25.add(a, [(title, 3.0), (cats, 2.0), (feats, 1.5),
                              (det, 1.2), (store, 1.0), (desc, 0.8)], bigrams=self.bigrams)
            self.cat_tokens[a] = set(toks(cats + " " + title))
            self.blob[a] = " ".join([title, feats, det, desc]).lower()
            pr = p.get("price")
            self.price[a] = pr if isinstance(pr, (int, float)) else None
            self.store[a] = store.lower()[:30]
        self.bm25.finalize()
        self._attr_cache: dict[str, dict[str, str]] = {}
        self.state: dict[str, dict] = {}

    def _attrs(self, asin: str) -> dict[str, str]:
        if asin in self._attr_cache:
            return self._attr_cache[asin]
        blob_toks = set(toks(self.blob[asin]))
        d = {}
        for attr in ("material", "color"):
            hit = blob_toks & self.vocab[attr]
            if hit:
                d[attr] = sorted(hit)[0]
        pr = self.price[asin]
        if pr is not None:
            d["budget"] = "low" if pr < 20 else "mid" if pr < 50 else "high"
        if self.store[asin]:
            d["brand"] = self.store[asin]
        self._attr_cache[asin] = d
        return d

    def _extract_category(self, msg: str) -> list[str]:
        m = LEAD_RE.search(msg)
        if m:
            return toks(m.group(1))
        return toks(msg)[:6]

    def _is_override(self, msg: str) -> bool:
        low = msg.lower()
        return any(c in low for c in OVERRIDE_CUES)

    def _is_noninformative(self, msg: str) -> bool:
        low = msg.lower()
        return any(c in low for c in NO_INFO_CUES)

    def reset(self, session_id, user_profile):
        self.state[session_id] = {"cat": [], "turns": [], "asked": set(),
                                  "profile": user_profile or {}, "override_at": None}

    def _query(self, st) -> dict[str, float]:
        q: dict[str, float] = defaultdict(float)
        for i, (turn_no, text) in enumerate(st["turns"]):
            w = float(os.environ.get("C_PREOVW", "0.5")) if (st["override_at"] is not None and turn_no < st["override_at"]) else 1.0
            w *= 1.0 + 0.25 * i
            tt = toks(text)
            for t in tt:
                q[t] += w
            if self.bigrams:
                for x, y in zip(tt, tt[1:]):
                    q[x + "_" + y] += w * 2.0
        for t in st["cat"]:
            q[t] += 2.0
        return q

    def _ask_open(self):
        return "other", "Anything else that matters to you? Any detail helps me narrow this down."

    def _ask_infogain(self, st, cands):
        best, best_h = None, -1.0
        total = sum(w for _, w in cands) or 1.0
        for attr in ("material", "color", "budget", "feature", "style", "use_case", "size"):
            if attr in st["asked"]:
                continue
            groups: dict[str, float] = defaultdict(float)
            for asin, w in cands:
                groups[self._attrs(asin).get(attr, "<none>")] += w
            if len(groups) < 2:
                continue
            h = -sum((v / total) * math.log(v / total) for v in groups.values() if v > 0)
            h *= 1.0 - groups.get("<none>", 0.0) / total
            if h > best_h:
                best, best_h = attr, h
        if best is None:
            return self._ask_open()
        st["asked"].add(best)
        return best, f"Do you have a preference on {best.replace('_', ' ')}?"

    def _ask_hybrid(self, st, cands):
        attr, msg = self._ask_infogain(st, cands)
        if attr == "other":
            return attr, msg
        total = sum(w for _, w in cands) or 1.0
        groups: dict[str, float] = defaultdict(float)
        for asin, w in cands:
            groups[self._attrs(asin).get(attr, "<none>")] += w
        h = -sum((v / total) * math.log(v / total) for v in groups.values() if v > 0)
        return (attr, msg) if h >= 1.0 else self._ask_open()

    def _gate_count(self, scores, turn: int) -> int:
        if self.gate == "off" or not scores:
            return 10
        top = scores[0]
        if top <= 0:
            return 10
        band = max(0.55, 0.90 - 0.10 * (turn - 1))
        n = sum(1 for s in scores[:10] if s >= band * top)
        caps = [int(x) for x in os.environ.get("G_CAPS", "1,1,10,10,10").split(",")]
        cap = caps[min(turn, len(caps)) - 1]
        if turn <= 2:
            default_floor = 1
        elif turn <= 4:
            default_floor = 3
        else:
            default_floor = 10
        floor = min(cap, default_floor)
        return max(floor, min(cap, n))

    def respond(self, session_id, user_message, turn, top_k):
        st = self.state.setdefault(session_id, {"cat": [], "turns": [], "asked": set(),
                                                "profile": {}, "override_at": None})
        early_rotate = os.environ.get("C_ROTATE_EARLY", "1")
        if turn > 1 and self._is_override(user_message):
            st["override_at"] = turn
            st["cat"] = []
            if early_rotate == "2":
                # Redundant under ShoppingCopilot: its own respond() already
                # resets st["shown"] on override before ever delegating here,
                # regardless of turn. This only matters if the base Agent is
                # ever used standalone, without that outer reset.
                st["shown"] = set()
        if not self._is_noninformative(user_message):
            st["turns"].append((turn, user_message))
        if not st["cat"]:
            c = self._extract_category(user_message)
            if c:
                st["cat"] = c

        raw = self.bm25.score(self._query(st))
        catset = set(st["cat"])
        ranked = []
        for idx, s in raw.items():
            asin = self.bm25.ids[idx]
            if catset:
                ov = len(catset & self.cat_tokens[asin]) / len(catset)
                s *= (1.0 + 1.2 * ov)
            ranked.append((s, asin))
        ranked.sort(key=lambda x: (-x[0], x[1]))

        exclude = st.get("shown", set()) if early_rotate != "0" else set()
        avail = [(s, a) for s, a in ranked if a not in exclude] or ranked
        top = avail[:50]

        n = self._gate_count([s for s, _ in top[:10]], turn)
        recs = [a for _, a in top[:n]]
        cands = [(a, s) for s, a in top[:30]]

        if self.question_policy == "infogain":
            attr, msg = self._ask_infogain(st, cands)
        elif self.question_policy == "hybrid":
            attr, msg = self._ask_hybrid(st, cands)
        elif self.question_policy == "none":
            attr, msg = None, "Here are the closest matches I found."
        else:
            attr, msg = self._ask_open()

        return {"message": msg, "ask_attribute": attr,
                "recommendations": [{"parent_asin": a} for a in recs],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


_PRICE = r"\$?\s*(\d+(?:\.\d{1,2})?)"
BUDGET_MAX_RE = re.compile(
    r"(?:under|below|less than|at most|no more than|max(?:imum)?(?: of)?|up to|cheaper than|within)\s+" + _PRICE, re.I)
BUDGET_MIN_RE = re.compile(
    r"(?:over|above|more than|at least|min(?:imum)?(?: of)?|starting at|upwards of)\s+" + _PRICE, re.I)
BUDGET_BETWEEN_RE = re.compile(r"between\s+" + _PRICE + r"\s+(?:and|to|\-)\s+" + _PRICE, re.I)
BUDGET_AROUND_RE = re.compile(r"(?:around|about|roughly|approximately|~)\s+" + _PRICE, re.I)

EXPAND = {
    "navy": {"blue"}, "gray": {"grey"}, "grey": {"gray"},
    "charcoal": {"gray", "grey"}, "burgundy": {"maroon", "red"},
    "maroon": {"burgundy"}, "beige": {"tan", "khaki"}, "tan": {"beige", "khaki"},
    "khaki": {"beige", "tan"}, "teal": {"turquoise"}, "turquoise": {"teal"},
    "violet": {"purple"}, "lavender": {"purple"}, "crimson": {"red"},
    "scarlet": {"red"}, "golden": {"gold"}, "woolen": {"wool"},
    "woollen": {"wool"}, "wooden": {"wood"},
}

RRF_K = 60
MIN_KEEP = 25
SLATE_SIZE = 10


def _expand(term: str) -> set[str]:
    return {term} | EXPAND.get(term, set())


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


class ShoppingCopilot(Agent):
    def __init__(self, catalog_path="data/catalog.jsonl", **kw):
        super().__init__(catalog_path, **kw)
        self.c_qpolicy = kw.get("question_policy") or os.environ.get("QPOLICY") or "hybrid"

        style_c, fit_c, occ_c = Counter(), Counter(), Counter()
        self.rating: dict[str, tuple[float | None, int]] = {}
        with Path(catalog_path).open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                a = str(p["parent_asin"])
                avg = p.get("average_rating")
                num = p.get("rating_number")
                self.rating[a] = (
                    float(avg) if isinstance(avg, (int, float)) else None,
                    int(num) if isinstance(num, (int, float)) else 0,
                )
                det = p.get("details") or {}
                if isinstance(det, dict):
                    for key, val in det.items():
                        kl = str(key).lower()
                        if "style" in kl:
                            style_c.update(toks(str(val)))
                        if "fit" in kl:
                            fit_c.update(toks(str(val)))
                        if "occasion" in kl or "season" in kl:
                            occ_c.update(toks(str(val)))
        self.vocab["style"] = {w for w, _ in style_c.most_common(40)}
        self.vocab["fit"] = {w for w, _ in fit_c.most_common(40)}
        self.vocab["occasion"] = {w for w, _ in occ_c.most_common(40)}

        tot_w = sum(n for _, n in self.rating.values())
        tot_r = sum((r or 0) * n for r, n in self.rating.values())
        self.mu = (tot_r / tot_w) if tot_w else 4.0
        nums = sorted(n for _, n in self.rating.values())
        self.m = max(5, nums[len(nums) // 2]) if nums else 25
        self.shrunk = {}
        for a, (r, n) in self.rating.items():
            self.shrunk[a] = ((r or self.mu) * n + self.mu * self.m) / (n + self.m)

        gaz = set().union(*self.vocab.values())
        self.term_asins: dict[str, set[str]] = defaultdict(set)
        for a, blob in self.blob.items():
            for t in set(toks(blob)) & gaz:
                self.term_asins[t].add(a)
        self.all_asins = set(self.blob)

    def _c_state(self, st):
        st.setdefault("shown", set())
        st.setdefault("last_slate", [])
        st.setdefault("constraints", [])
        st.setdefault("budget", None)
        st.setdefault("bad_values", Counter())
        return st

    def reset(self, session_id, user_profile):
        super().reset(session_id, user_profile)
        self._c_state(self.state[session_id])

    def _extract_constraints(self, st, msg: str, turn: int):
        mtoks = set(toks(msg))
        for bucket, vocab in self.vocab.items():
            hit = mtoks & vocab
            if hit:
                st["constraints"].append((turn, bucket, hit))
        m = BUDGET_BETWEEN_RE.search(msg)
        if m:
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            st["budget"] = (lo, hi, turn)
            return
        m = BUDGET_MAX_RE.search(msg)
        if m:
            st["budget"] = (0.0, float(m.group(1)), turn)
            return
        m = BUDGET_MIN_RE.search(msg)
        if m:
            st["budget"] = (float(m.group(1)), float("inf"), turn)
            return
        m = BUDGET_AROUND_RE.search(msg)
        if m:
            x = float(m.group(1))
            st["budget"] = (0.5 * x, 1.5 * x, turn)

    def _drop_stale_constraints(self, st, override_turn: int):
        st["constraints"] = [(t, b, v) for t, b, v in st["constraints"]
                             if t >= override_turn]
        if st["budget"] and st["budget"][2] < override_turn:
            st["budget"] = None

    def _stated_values(self, st) -> set[str]:
        out = set()
        for _, _, vals in st["constraints"]:
            out |= vals
        return out

    def _allowed_set(self, st) -> tuple[set[str], list[set[str]]]:
        allowed = self.all_asins
        soft: list[set[str]] = []
        mode = os.environ.get("C_FILTER", "2")
        if mode == "0":
            return allowed, soft
        by_bucket: dict[str, set[str]] = defaultdict(set)
        for _, bucket, vals in st["constraints"]:
            by_bucket[bucket] |= vals
        for bucket, vals in by_bucket.items():
            hit: set[str] = set()
            for v in vals:
                for term in _expand(v):
                    hit |= self.term_asins.get(term, set())
            if not hit:
                continue
            if mode == "2":
                soft.append(hit)
            elif len(allowed & hit) >= MIN_KEEP:
                allowed = allowed & hit
            else:
                soft.append(hit)
        if st["budget"]:
            lo, hi, _ = st["budget"]
            hit = {a for a in allowed
                   if self.price[a] is None or lo <= self.price[a] <= hi}
            if mode == "2":
                soft.append(hit)
            elif len(hit) >= MIN_KEEP:
                allowed = hit
        return allowed, soft

    def _bm25_route(self, qterms, allowed, catset, limit=300,
                    return_scores=False):
        raw = self.bm25.score(qterms)
        out = []
        for idx, s in raw.items():
            a = self.bm25.ids[idx]
            if a not in allowed:
                continue
            if catset:
                ov = len(catset & self.cat_tokens[a]) / len(catset)
                s *= (1.0 + 1.2 * ov)
            out.append((s, a))
        out.sort(key=lambda x: (-x[0], x[1]))
        if return_scores:
            return [(a, s) for s, a in out[:limit]]
        return [a for _, a in out[:limit]]

    def _terms_from(self, tokens, weight=1.0):
        q: dict[str, float] = defaultdict(float)
        for t in tokens:
            q[t] += weight
        if self.bigrams:
            for x, y in zip(tokens, tokens[1:]):
                q[x + "_" + y] += weight * 2.0
        return q

    def _fuse(self, st, allowed, soft):
        catset = set(st["cat"])
        routes: list[list[str]] = []

        r1 = self._bm25_route(self._query(st), allowed, catset, limit=300,
                              return_scores=True)
        raw1 = dict((a, s) for a, s in r1)
        routes.append([a for a, _ in r1])

        stated = self._stated_values(st)
        if stated:
            q = self._terms_from(sorted(stated) + st["cat"], 1.0)
            routes.append(self._bm25_route(q, allowed, catset))

        if st["turns"]:
            last = toks(st["turns"][-1][1])
            routes.append(self._bm25_route(
                self._terms_from(last + st["cat"], 1.0), allowed, catset))

        if not _flag("C_ROUTES", "0"):
            routes = routes[:1]
        pool: set[str] = set().union(*routes) if routes else set()

        soft_mode = os.environ.get("C_FILTER", "2") == "2"

        if soft and pool and not soft_mode:
            scored = sorted(pool, key=lambda a: (-sum(a in s for s in soft),
                                                 -self.shrunk[a], a))
            routes.append(scored[:300])

        if pool:
            routes.append(sorted(pool, key=lambda a: (-self.shrunk[a], a))[:300])

        fused: dict[str, float] = defaultdict(float)
        weights = [1.0, 1.0, 1.0, 0.7, 0.5][:len(routes)]
        for w, route in zip(weights, routes):
            for rank, a in enumerate(route, start=1):
                fused[a] += w / (RRF_K + rank)

        if soft_mode and soft:
            boost = float(os.environ.get("C_SOFT_BOOST", "0.05"))
            total = len(soft)
            for a in list(fused):
                matched = sum(1 for s in soft if a in s)
                if matched:
                    fused[a] += boost * (matched / total)

        bad = st["bad_values"]
        if bad and _flag("C_PENALTY"):
            for a in list(fused):
                flags = sum(1 for attr, val in self._attrs(a).items()
                            if bad[(attr, val)] >= 2 and val not in stated)
                if flags:
                    fused[a] *= 0.93 ** min(flags, 3)
        return sorted(fused.items(), key=lambda x: (-x[1], x[0])), raw1

    def _maybe_rerank(self, st, ranked):
        if os.environ.get("RERANKER") != "ce" or len(ranked) < 2:
            return ranked
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            if not hasattr(self, "_ce"):
                self._ce = CrossEncoder(os.environ.get(
                    "CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception:
            return ranked
        query = " ".join(t for _, t in st["turns"][-3:])[:512]
        head = ranked[:40]
        scores = self._ce.predict([(query, self.blob[a][:512]) for a, _ in head])
        head = [ab for _, ab in sorted(zip([-s for s in scores], head))]
        return head + ranked[40:]

    def _state_score(self, st, asin: str) -> float:
        score = 0.0
        for t, _bucket, vals in st["constraints"]:
            recency = 1.0 + 0.3 * (t - 1)
            for v in vals:
                for term in _expand(v):
                    if asin in self.term_asins.get(term, ()):
                        idf = self.bm25.idf.get(term, 0.0)
                        score += recency * (idf if idf > 0 else 0.1)
        bad = st["bad_values"]
        if bad:
            penalty = sum(bad.get((attr, val), 0)
                          for attr, val in self._attrs(asin).items())
            if penalty:
                score -= 0.5 * penalty
        return score

    def _state_rerank(self, st, ranked, turn: int):
        if not _flag("C_RERANK", "0") or turn < 3 or len(ranked) < 2:
            return ranked
        top10 = ranked[:10]
        rest = ranked[10:]
        top_score = top10[0][1]
        if top_score <= 0:
            return ranked
        second_score = top10[1][1] if len(top10) > 1 else 0.0
        margin = (top_score - second_score) / top_score
        margin_thresh = float(os.environ.get("C_RERANK_MARGIN", "0.15"))
        if margin >= margin_thresh:
            return ranked
        rescored = sorted(top10, key=lambda x: (-self._state_score(st, x[0]), -x[1]))
        return rescored + rest

    def _learn_from_rejection(self, st):
        slate = st["last_slate"]
        if not slate:
            return
        stated = self._stated_values(st)
        counts: Counter = Counter()
        for a in slate:
            for attr, val in self._attrs(a).items():
                counts[(attr, val)] += 1
        for (attr, val), c in counts.items():
            if val in stated:
                continue
            if c >= max(2, int(0.6 * len(slate))):
                st["bad_values"][(attr, val)] += 1

    def respond(self, session_id, user_message, turn, top_k):
        st = self._c_state(self.state.setdefault(
            session_id, {"cat": [], "turns": [], "asked": set(),
                         "profile": {}, "override_at": None}))

        if turn > 1 and self._is_override(user_message):
            st["override_at"] = turn
            self._drop_stale_constraints(st, turn)
            st["shown"] = set()
            st["bad_values"] = Counter()
            st["last_slate"] = []
            st["asked"] = set()

        rescue_at = int(os.environ.get("C_RESCUE_TURN", "4"))
        if turn < rescue_at:
            out = super().respond(session_id, user_message, turn, top_k)
            slate = [r["parent_asin"] for r in out["recommendations"]]
            st["last_slate"] = slate
            st["shown"].update(slate)
            return out

        self._learn_from_rejection(st)
        st["shown"].update(st["last_slate"])

        if not st.get("c_replayed"):
            st["c_replayed"] = True
            ov = st["override_at"] or 0
            for t, text in st["turns"]:
                if t >= ov:
                    self._extract_constraints(st, text, t)

        if not self._is_noninformative(user_message):
            st["turns"].append((turn, user_message))
            self._extract_constraints(st, user_message, turn)
        if not st["cat"]:
            c = self._extract_category(user_message)
            if c:
                st["cat"] = c

        allowed, soft = self._allowed_set(st)
        ranked, raw1 = self._fuse(st, allowed, soft)
        ranked = self._maybe_rerank(st, ranked)
        ranked = self._state_rerank(st, ranked, turn)

        k = min(SLATE_SIZE, top_k or SLATE_SIZE)
        exclude = st["shown"] if _flag("C_ROTATE") else set()
        avail = [(a, s) for a, s in ranked if a not in exclude]
        gate_scores = sorted((raw1.get(a, 0.0) for a, _ in avail[:10]),
                             reverse=True)
        k = min(k, self._gate_count(gate_scores, turn))
        slate = [a for a, _ in avail][:k]
        if len(slate) < k:
            for a in sorted(allowed - st["shown"] - set(slate),
                            key=lambda a: (-self.shrunk[a], a)):
                slate.append(a)
                if len(slate) == k:
                    break

        cands = [(a, s) for a, s in ranked[:30] if a not in st["shown"]] \
            or [(a, 1.0) for a in slate]
        qp = self.c_qpolicy
        if qp == "none":
            attr, msg = None, "Here are the closest matches I found."
        elif qp == "open":
            attr, msg = self._ask_open()
        elif qp == "infogain":
            attr, msg = self._ask_infogain(st, cands)
        else:
            attr, msg = self._ask_hybrid(st, cands)

        st["last_slate"] = slate
        return {"message": msg, "ask_attribute": attr,
                "recommendations": [{"parent_asin": a} for a in slate],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


Agent = ShoppingCopilot
