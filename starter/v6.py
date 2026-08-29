
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "for", "something", "need", "like", "prefer", "find", "show",
}

BUYING_WORDS = {
    "buy", "purchase", "need", "want", "looking", "require", "required",
    "must", "under", "below", "budget", "size", "color", "colour",
    "material", "price", "cheap", "affordable", "work", "running",
    "hiking", "gym", "winter",
}

BROWSING_WORDS = {
    "exploring", "explore", "browse", "browsing", "ideas", "options",
    "suggest", "suggestions", "recommend", "recommendations", "similar",
    "interesting", "maybe", "perhaps", "looking", "discover",
}


def _text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        return " ".join(
            f"{key} {item}"
            for key, item in value.items()
        )

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []

    return [
        " ".join(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


class Agent:
    """
    V6.1

    Fast hybrid conversational retrieval.

    Components:
    - Stateful conversation memory
    - Buying / browsing intent routing
    - Weighted BM25 over catalog fields
    - Exact phrase matching
    - 2-gram / 3-gram matching
    - Lightweight category matching
    - Conservative rating prior
    - Reciprocal Rank Fusion
    - Lightweight MMR diversity
    - Improved intent-override handling
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> None:

        self.catalog_path = Path(catalog_path)

        self.connection = sqlite3.connect(":memory:")

        self._sessions: dict[str, dict] = {}

        self._metadata: dict[str, dict] = {}

        self._build_index()

    # =========================================================
    # INDEX
    # =========================================================

    def _build_index(self) -> None:

        cursor = self.connection.cursor()

        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, "
            "title, "
            "categories, "
            "features, "
            "details, "
            "store, "
            "description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch = []

        with self.catalog_path.open(
            encoding="utf-8"
        ) as handle:

            for line in handle:

                product = json.loads(line)

                asin = str(product["parent_asin"])

                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))

                batch.append(
                    (
                        asin,
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
                    )
                )

                # Lightweight metadata used by the final
                # ranking stage.
                rating = product.get("average_rating")

                try:
                    rating = float(rating)
                except (TypeError, ValueError):
                    rating = 0.0

                self._metadata[asin] = {
                    "title": title.lower(),
                    "categories": categories.lower(),
                    "features": features.lower(),
                    "details": details.lower(),
                    "store": store.lower(),
                    "description": description.lower(),
                    "rating": rating,
                }

                if len(batch) >= 1000:

                    cursor.executemany(
                        "INSERT INTO products VALUES "
                        "(?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )

                    batch.clear()

        if batch:

            cursor.executemany(
                "INSERT INTO products VALUES "
                "(?, ?, ?, ?, ?, ?, ?)",
                batch,
            )

        self.connection.commit()

    # =========================================================
    # SESSION
    # =========================================================

    def reset(
        self,
        session_id: str,
        user_profile: dict,
    ) -> None:

        self._sessions[session_id] = {
            "user_profile": user_profile or {},
            "history": [],
            "intent": None,
            "override_active": False,
        }

    # =========================================================
    # INTENT DETECTION
    # =========================================================

    def _detect_intent(self, text: str) -> str:

        tokens = set(_terms(text))

        buying_score = len(
            tokens & BUYING_WORDS
        )

        browsing_score = len(
            tokens & BROWSING_WORDS
        )

        if buying_score > browsing_score:
            return "buying"

        if browsing_score > buying_score:
            return "browsing"

        attribute_words = {
            "black",
            "white",
            "blue",
            "red",
            "green",
            "brown",
            "cotton",
            "leather",
            "wool",
            "nylon",
            "polyester",
            "small",
            "medium",
            "large",
        }

        if tokens & attribute_words:
            return "buying"

        return "browsing"

    # =========================================================
    # QUERY BUILDING
    # =========================================================

    def _build_query(
        self,
        history: list[str],
    ) -> tuple[str, list[str]]:

        conversation = " ".join(history)

        tokens = list(
            dict.fromkeys(
                _terms(conversation)
            )
        )

        # Keep the query compact.
        tokens = tokens[-80:]

        if not tokens:
            return "", []

        expression = " OR ".join(
            f'"{term}"'
            for term in tokens
        )

        return expression, tokens

    # =========================================================
    # BM25 RETRIEVAL
    # =========================================================

    def _bm25_candidates(
        self,
        expression: str,
        limit: int = 80,
        intent: str = "buying",
    ) -> list[str]:

        if not expression:
            return []

        if intent == "buying":

            weights = (
                8.0,   # title
                3.0,   # categories
                6.0,   # features
                3.0,   # details
                1.5,   # store
                2.0,   # description
            )

        else:

            weights = (
                5.0,   # title
                5.0,   # categories
                5.0,   # features
                3.0,   # details
                1.5,   # store
                3.0,   # description
            )

        rows = self.connection.execute(
            "SELECT parent_asin "
            "FROM products "
            "WHERE products MATCH ? "
            "ORDER BY bm25("
            "products, "
            "?, ?, ?, ?, ?, ?"
            ") "
            "LIMIT ?",
            (
                expression,
                *weights,
                limit,
            ),
        ).fetchall()

        return [
            str(row[0])
            for row in rows
        ]

    # =========================================================
    # EXACT PHRASE + N-GRAM RETRIEVAL
    # =========================================================

    def _phrase_candidates(
        self,
        tokens: list[str],
        limit: int = 80,
    ) -> list[str]:

        phrases = []

        phrases.extend(
            _ngrams(tokens, 3)
        )

        phrases.extend(
            _ngrams(tokens, 2)
        )

        # Prevent an excessive number of FTS queries.
        phrases = phrases[-20:]

        scores: Counter[str] = Counter()

        for phrase in phrases:

            expression = (
                '"'
                + phrase.replace('"', "")
                + '"'
            )

            try:

                rows = self.connection.execute(
                    "SELECT parent_asin "
                    "FROM products "
                    "WHERE products MATCH ? "
                    "LIMIT ?",
                    (
                        expression,
                        limit,
                    ),
                ).fetchall()

            except sqlite3.Error:
                continue

            for rank, row in enumerate(
                rows,
                start=1,
            ):

                asin = str(row[0])

                scores[asin] += (
                    1.0 / rank
                )

        return [
            asin
            for asin, _ in scores.most_common(
                limit
            )
        ]

    # =========================================================
    # CATEGORY RETRIEVAL
    # =========================================================

    def _category_candidates(
        self,
        tokens: list[str],
        limit: int = 80,
    ) -> list[str]:

        if not tokens:
            return []

        category_terms = tokens[-20:]

        expression = " OR ".join(
            f'"{term}"'
            for term in category_terms
        )

        rows = self.connection.execute(
            "SELECT parent_asin "
            "FROM products "
            "WHERE categories MATCH ? "
            "LIMIT ?",
            (
                expression,
                limit,
            ),
        ).fetchall()

        return [
            str(row[0])
            for row in rows
        ]

    # =========================================================
    # RATING PRIOR
    # =========================================================

    def _rating_score(
        self,
        asin: str,
    ) -> float:

        metadata = self._metadata.get(
            asin
        )

        if not metadata:
            return 0.0

        rating = metadata["rating"]

        if rating <= 0:
            return 0.0

        # Conservative Bayesian shrinkage.
        prior = 4.0
        prior_strength = 5.0
        pseudo_count = 5.0

        posterior = (
            rating * pseudo_count
            + prior * prior_strength
        ) / (
            pseudo_count
            + prior_strength
        )

        return max(
            0.0,
            min(
                1.0,
                posterior / 5.0,
            ),
        )

    # =========================================================
    # RECIPROCAL RANK FUSION
    # =========================================================

    def _rrf(
        self,
        routes: list[list[str]],
        weights: list[float],
    ) -> dict[str, float]:

        scores: dict[str, float] = {}

        k = 60.0

        for route, weight in zip(
            routes,
            weights,
        ):

            for rank, asin in enumerate(
                route,
                start=1,
            ):

                scores[asin] = (
                    scores.get(asin, 0.0)
                    + weight / (k + rank)
                )

        return scores

    # =========================================================
    # LIGHTWEIGHT RERANKING
    # =========================================================

    def _rerank(
        self,
        candidates: list[str],
        tokens: list[str],
        intent: str,
        top_k: int,
    ) -> list[str]:

        query_text = " ".join(tokens)

        query_2grams = set(
            _ngrams(tokens, 2)
        )

        query_3grams = set(
            _ngrams(tokens, 3)
        )

        scored = []

        for base_rank, asin in enumerate(
            candidates
        ):

            metadata = self._metadata.get(
                asin
            )

            if not metadata:
                continue

            title = metadata["title"]
            features = metadata["features"]
            categories = metadata["categories"]

            combined = (
                title
                + " "
                + features
                + " "
                + categories
            )

            score = 0.0

            # -------------------------------------------------
            # Full exact phrase
            # -------------------------------------------------

            if (
                query_text
                and query_text in combined
            ):
                score += 0.12

            # Exact phrase in title is more valuable.
            if (
                query_text
                and query_text in title
            ):
                score += 0.20

            # -------------------------------------------------
            # 3-gram matches
            # -------------------------------------------------

            for phrase in query_3grams:

                if phrase in combined:
                    score += 0.045

                if phrase in title:
                    score += 0.075

            # -------------------------------------------------
            # 2-gram matches
            # -------------------------------------------------

            for phrase in query_2grams:

                if phrase in combined:
                    score += 0.025

                if phrase in title:
                    score += 0.045

            # -------------------------------------------------
            # Term coverage
            # -------------------------------------------------

            matched = sum(
                1
                for token in tokens
                if token in combined
            )

            if tokens:

                coverage = (
                    matched / len(tokens)
                )

                score += (
                    0.08 * coverage
                )

            # -------------------------------------------------
            # Rating prior
            # -------------------------------------------------

            score += (
                0.015
                * self._rating_score(asin)
            )

            # Preserve small amount of RRF rank influence.
            score += (
                1.0
                / (1000.0 + base_rank)
            )

            scored.append(
                (
                    score,
                    asin,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            asin
            for _, asin in scored[
                :max(top_k * 4, 20)
            ]
        ]

    # =========================================================
    # MMR
    # =========================================================

    def _mmr(
        self,
        candidates: list[str],
        tokens: list[str],
        top_k: int,
        intent: str,
    ) -> list[str]:

        if not candidates:
            return []

        selected: list[str] = []

        query_set = set(tokens)

        if intent == "buying":
            lambda_value = 0.88
        else:
            lambda_value = 0.72

        candidate_sets: dict[
            str,
            set[str],
        ] = {}

        for asin in candidates:

            metadata = self._metadata.get(
                asin,
                {},
            )

            text = (
                metadata.get("title", "")
                + " "
                + metadata.get("features", "")
                + " "
                + metadata.get("categories", "")
            )

            candidate_sets[asin] = set(
                _terms(text)
            )

        remaining = list(candidates)

        while (
            remaining
            and len(selected) < top_k
        ):

            best_asin = None
            best_score = -float("inf")

            for asin in remaining:

                relevance = len(
                    query_set
                    & candidate_sets[asin]
                )

                if query_set:
                    relevance /= len(
                        query_set
                    )

                if not selected:

                    diversity_penalty = 0.0

                else:

                    similarities = []

                    for chosen in selected:

                        a = candidate_sets[asin]
                        b = candidate_sets[chosen]

                        union = len(a | b)

                        if union == 0:
                            similarity = 0.0
                        else:
                            similarity = (
                                len(a & b)
                                / union
                            )

                        similarities.append(
                            similarity
                        )

                    diversity_penalty = max(
                        similarities
                    )

                mmr_score = (
                    lambda_value
                    * relevance
                    - (
                        1.0
                        - lambda_value
                    )
                    * diversity_penalty
                )

                if mmr_score > best_score:

                    best_score = mmr_score
                    best_asin = asin

            if best_asin is None:
                break

            selected.append(
                best_asin
            )

            remaining.remove(
                best_asin
            )

        return selected

    # =========================================================
    # RESPONSE
    # =========================================================

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:

        if session_id not in self._sessions:

            raise RuntimeError(
                "reset must be called before respond"
            )

        state = self._sessions[
            session_id
        ]

        lower_message = (
            user_message.lower()
        )

        # -----------------------------------------------------
        # Detect evaluator intent override.
        # -----------------------------------------------------

        is_override = (
            "ignore my earlier" in lower_message
            or "ignore earlier" in lower_message
            or "actually" in lower_message
        )

        if is_override:

            # IMPORTANT:
            # Do NOT delete the previous conversation.
            #
            # The new message should override the old
            # preference while preserving the original
            # product/category context.
            state["override_active"] = True

        state["history"].append(
            user_message
        )

        # -----------------------------------------------------
        # Query construction
        # -----------------------------------------------------

        if state.get(
            "override_active",
            False,
        ):

            # Keep all previous context but repeat the newest
            # message so its constraints receive substantially
            # greater weight in BM25 and downstream ranking.
            history_for_query = (
                state["history"][:-1]
                + [
                    state["history"][-1]
                ] * 4
            )

        else:

            history_for_query = (
                state["history"]
            )

        conversation = " ".join(
            history_for_query
        )

        intent = self._detect_intent(
            conversation
        )

        state["intent"] = intent

        expression, tokens = (
            self._build_query(
                history_for_query
            )
        )

        if not expression:

            recommendations = []

        else:

            # -------------------------------------------------
            # Retrieval routes
            # -------------------------------------------------

            bm25_route = (
                self._bm25_candidates(
                    expression,
                    limit=80,
                    intent=intent,
                )
            )

            phrase_route = (
                self._phrase_candidates(
                    tokens,
                    limit=80,
                )
            )

            category_route = (
                self._category_candidates(
                    tokens,
                    limit=80,
                )
            )

            # -------------------------------------------------
            # Route weighting
            # -------------------------------------------------

            if intent == "buying":

                routes = [
                    bm25_route,
                    phrase_route,
                    category_route,
                ]

                weights = [
                    0.62,
                    0.28,
                    0.10,
                ]

            else:

                routes = [
                    bm25_route,
                    category_route,
                    phrase_route,
                ]

                weights = [
                    0.50,
                    0.30,
                    0.20,
                ]

            # -------------------------------------------------
            # Reciprocal Rank Fusion
            # -------------------------------------------------

            rrf_scores = self._rrf(
                routes,
                weights,
            )

            candidates = [
                asin
                for asin, _ in sorted(
                    rrf_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:100]
            ]

            # -------------------------------------------------
            # Lightweight relevance reranking
            # -------------------------------------------------

            reranked = self._rerank(
                candidates,
                tokens,
                intent,
                top_k,
            )

            # -------------------------------------------------
            # MMR slate diversification
            # -------------------------------------------------

            recommendations = self._mmr(
                reranked,
                tokens,
                top_k,
                intent,
            )

        return {
            "message": (
                "Here are the closest matches I found."
            ),
            "ask_attribute": None,
            "recommendations": [
                {
                    "parent_asin": asin
                }
                for asin in recommendations[:top_k]
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }

