"""Catalog-derived dense retrieval (LSA) -- Pillar I semantic route.

No pretrained weights, no network, no vector database: the semantic space is
factorised from the frozen catalog's own text at index-build time with a fixed
random_state, so the vectors are a deterministic function of catalog.jsonl
alone. Cosine similarity in that space supplies the "semantic" leg of
multi-route retrieval that lexical BM25 cannot: it matches a vague browsing
query to products whose wording never overlaps it.

Used as a *blend* over an existing candidate list -- it reorders, it never
filters. A candidate BM25 surfaced can be demoted but never dropped, so
hit-rate@10 is structurally protected against a bad semantic score.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 20260831


def build(ids, docs, dims=192, cache=None):
    """Build (or load) the LSA index. The cache is a pure speed-up: the
    vectors it holds are the same deterministic function of the catalog."""
    import os
    if cache and os.path.exists(cache):
        import joblib
        return joblib.load(cache)
    idx = DenseIndex(ids, docs, dims=dims)
    if cache:
        import joblib
        joblib.dump(idx, cache, compress=0)
    return idx


class DenseIndex:
    def __init__(self, ids, docs, dims=192, min_df=2, max_features=120000):
        self.ids = list(ids)
        self.pos = {a: i for i, a in enumerate(self.ids)}
        self.vec = TfidfVectorizer(lowercase=True, sublinear_tf=True,
                                   min_df=min_df, max_features=max_features,
                                   ngram_range=(1, 2), dtype=np.float32)
        tfidf = self.vec.fit_transform(docs)
        k = min(dims, min(tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=k, algorithm="randomized",
                                n_iter=5, random_state=SEED)
        self.emb = normalize(self.svd.fit_transform(tfidf).astype(np.float32),
                             copy=False)

    def query_vec(self, text):
        q = self.svd.transform(self.vec.transform([text]))
        return normalize(q.astype(np.float32), copy=False)[0]

    def cosines(self, text, asins):
        idx = [self.pos[a] for a in asins if a in self.pos]
        if not idx:
            return {}
        sims = self.emb[idx] @ self.query_vec(text)
        out, j = {}, 0
        for a in asins:
            if a in self.pos:
                out[a] = round(float(sims[j]), 6); j += 1
        return out
