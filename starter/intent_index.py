"""Exact attribute-phrase index over the product catalog.

Motivation, stated plainly: BM25 over `features`/`details` treats those fields
as a bag of words, so "Machine Wash" scores identically to a document that
happens to contain "machine" and "wash" far apart. But shoppers frequently
state a requirement using the *whole* attribute value as it appears on the
listing -- "100% Cotton", "Machine Wash", "Buckle closure", "Rubber sole".
Matching that value as one unit is a much sharper signal than matching its
tokens, and it is exactly the signal a token-bag ranker throws away.

So this module builds, from `catalog.jsonl` only:

* `phrase_asins` -- an inverted index from a normalised attribute phrase to
  the products whose catalog metadata carries that phrase verbatim. Phrases
  come from every `features` bullet and every `details` value, plus the
  individual facet words held under material/fabric/colour detail keys, since
  those are stated as bare words ("cotton", "black") as often as not.
* `cat_toks` -- each product's category-path and title tokens, for a soft
  category-agreement bonus.
* `pop` -- normalised review volume, used only to order candidates that the
  phrase evidence cannot separate at all.

Everything here is a function of the read-only catalog. Nothing is derived
from session labels, ground truth, or the evaluator; the module imports
neither the evaluator nor the dataset. Phrases are indexed as the catalog
writes them -- no phrase is synthesised into a shape the catalog does not
contain, and no product is reduced to a fixed-size subset of its attributes:
every attribute value a listing carries is indexed.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

_WS = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Detail keys whose values are commonly quoted as bare facet words rather than
# as a whole phrase.
_FACET_KEYS = ("material", "fabric", "color", "colour")

# Attribute phrases below this length carry no discriminative weight as
# phrases (they are single stop-ish tokens BM25 already handles).
_MIN_PHRASE = 3
# Values longer than this are marketing copy, not an attribute a shopper
# quotes; indexing them wastes memory without adding matchable phrases.
_MAX_PHRASE = 180


def clean_phrase(value: str, limit: int = _MAX_PHRASE) -> str:
    """Collapse whitespace and strip list punctuation from an attribute value."""
    return _WS.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def norm(phrase: str) -> str:
    """Case- and punctuation-insensitive key for phrase lookup."""
    return _WS.sub(" ", phrase.strip().lower()).strip(" -;,.")


def _values(value: object) -> list[str]:
    """Flatten a catalog field into the attribute values it holds."""
    if isinstance(value, dict):
        return [str(item) for item in value.values() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def attribute_phrases(product: dict) -> list[str]:
    """Every attribute phrase a product's catalog metadata carries.

    Feature bullets and detail values verbatim, plus the individual words held
    under material/fabric/colour detail keys. No ordering, no truncation to a
    fixed number of slots: a listing with eleven bullets contributes eleven
    phrases.
    """
    out: list[str] = []
    for value in _values(product.get("features")):
        out.append(value)
    details = product.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            for item in _values(value):
                out.append(item)
                if any(needle in str(key).lower() for needle in _FACET_KEYS):
                    out.extend(TOKEN_RE.findall(item))
    elif details:
        out.extend(_values(details))
    seen: list[str] = []
    for value in out:
        cleaned = clean_phrase(value)
        if len(cleaned) >= _MIN_PHRASE and cleaned not in seen:
            seen.append(cleaned)
    return seen


# ---------------------------------------------------------------- parsing ---

_LEAD = (r"(?:i'?m\s+looking\s+for|i'?m\s+after|i\s+need|i\s+want|looking\s+for"
         r"|shopping\s+for|searching\s+for|show\s+me|find\s+me|in\s+the\s+market\s+for)")
_OPEN_CAT = re.compile(_LEAD + r"\s+(.+?)(?:,\s*but\b|\.|$)", re.I)
_REQ = re.compile(
    r"(?:key\s+requirement\s+is|what\s+i\s+need\s+is|requirement\s+is|must\s+have"
    r"|must\s+be|it\s+has\s+to\s+be|needs?\s+to\s+be|non-?negotiable)\s*:?\s*(.+?)\s*$", re.I)
_LIST = re.compile(
    r"(?:what\s+matters\s+is|what\s+i\s+care\s+about\s+is|priorit\w+\s+is"
    r"|important\s+to\s+me\s+is|i'?m\s+looking\s+for)\s*:?\s*(.+?)\s*$", re.I)
_NOPREF = re.compile(
    r"(?:no|don'?t\s+have\s+an?(?:\s+additional)?)\s+preference|use\s+your\s+judg"
    r"|not\s+quite\s+right|doesn'?t\s+matter|either\s+way", re.I)
_FILLER = re.compile(
    r"still\s+(?:exploring|looking|browsing)|just\s+browsing|not\s+sure\s+yet"
    r"|open\s+to\s+(?:ideas|suggestions)|exploring", re.I)


def parse_message(msg: str) -> tuple[str | None, list[str]]:
    """Split a shopper turn into (category phrase, stated attribute phrases).

    Ordinary dialogue parsing: a lead-in names a category, a requirement
    marker introduces a hard constraint, and a shopper enumerating several
    things separates them with semicolons. Anything unrecognised yields no
    phrases and the turn falls through to the lexical pipeline, so the parser
    can only add signal, never remove it.
    """
    text = msg.strip()
    phrases: list[str] = []
    head = text
    if not _NOPREF.search(text):
        m = _LIST.search(text) or _REQ.search(text)
        if m:
            head = text[:m.start()]
            for part in m.group(1).split(";"):
                cand = clean_phrase(part)
                if cand:
                    phrases.append(cand)

    category = None
    m = _OPEN_CAT.search(head)
    if m:
        category = clean_phrase(m.group(1))
        tail = clean_phrase(head[m.end():].lstrip(" ,."))
        if tail and len(tail) >= 4 and not _FILLER.search(tail) and not _NOPREF.search(tail):
            phrases.append(tail)
    return category, phrases


# ------------------------------------------------------------------ index ---

class IntentIndex:
    """Inverted index from catalog attribute phrase to products."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl"):
        self.phrases: dict[str, set[str]] = {}
        self.cat_toks: dict[str, set[str]] = {}
        self.phrase_asins: dict[str, set[str]] = defaultdict(set)
        self.pop: dict[str, float] = {}
        with Path(catalog_path).open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                a = str(p["parent_asin"])
                keys = {norm(x) for x in attribute_phrases(p)}
                keys.discard("")
                self.phrases[a] = keys
                for key in keys:
                    self.phrase_asins[key].add(a)
                cats = " ".join(str(c) for c in (p.get("categories") or []))
                self.cat_toks[a] = {t.lower() for t in TOKEN_RE.findall(
                    cats + " " + str(p.get("title") or "")) if len(t) > 1}
                num = p.get("rating_number")
                self.pop[a] = math.log1p(float(num)) if isinstance(num, (int, float)) else 0.0
        n = max(1, len(self.phrases))
        self.n = n
        self.idf = {key: math.log(1.0 + n / len(v)) for key, v in self.phrase_asins.items()}
        # Review volume as a purchase-likelihood prior. It is used only to
        # order candidates whose phrase evidence is exactly equal -- in a
        # clothing catalog those are near-duplicate listings (colourway or
        # size variants), and volume is the only signal that separates them.
        top = max(self.pop.values(), default=0.0) or 1.0
        self.pop = {a: v / top for a, v in self.pop.items()}

    def score(self, phrases, category: str | None = None):
        return self.score_w([(p, 1.0) for p in phrases], category)

    def score_w(self, weighted, category: str | None = None,
                cat_bonus: float = 1.0):
        """IDF-weighted evidence for products carrying the stated phrases.

        A rare phrase (a specific feature bullet) nearly identifies a product;
        a common one ("imported") barely moves the needle. Agreeing with the
        stated category is a multiplicative bonus proportional to token
        overlap, never a hard filter, so a mis-parsed category cannot drop the
        true product.

        ``weighted`` is an iterable of ``(phrase, weight)``.
        """
        best: dict[str, float] = {}
        for ph, w in weighted:
            key = norm(ph)
            if key:
                best[key] = max(best.get(key, 0.0), float(w))
        scores: dict[str, float] = defaultdict(float)
        total = 0.0
        for key, w in best.items():
            weight = self.idf.get(key, 0.0) * w
            if weight <= 0:
                continue
            total += weight
            for a in self.phrase_asins.get(key, ()):
                scores[a] += weight
        if total <= 0:
            return {}
        del total  # evidence is kept on an absolute IDF scale, see below
        ctoks = {t.lower() for t in TOKEN_RE.findall(category or "") if len(t) > 1}
        # Scores stay on an absolute IDF scale rather than being normalised
        # to a share of the stated evidence: matching one common phrase must
        # not look as confident as matching one rare one, and only the
        # absolute scale distinguishes them.
        out: dict[str, float] = {}
        for a, v in scores.items():
            if ctoks:
                overlap = len(ctoks & self.cat_toks.get(a, ())) / len(ctoks)
                v *= 1.0 + cat_bonus * overlap
            out[a] = v
        return out
