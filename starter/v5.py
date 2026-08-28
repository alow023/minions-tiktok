from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

SEARCH_FIELDS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
}

COLORS = {
    "black", "white", "blue", "red", "pink", "green",
    "brown", "gray", "grey", "purple", "yellow", "orange",
}

USE_CASES = {
    "hiking", "running", "gym", "winter", "outdoor", "work",
}

OVERRIDE_PATTERNS = (
    "ignore my earlier",
    "ignore my previous",
    "ignore what i said",
    "forget my earlier",
    "forget my previous",
    "changed my mind",
    "what i need is",
    "actually,",
    "actually ",
    "instead",
    "rather",
)


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


def _ngrams(
    tokens: list[str],
    minimum: int = 2,
    maximum: int = 3,
) -> set[str]:

    result: set[str] = set()

    for n in range(minimum, maximum + 1):

        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            result.add(
                " ".join(tokens[i:i + n])
            )

    return result


class Agent:
    """
    V5

    Stateful hybrid conversational retrieval.

    Retrieval pipeline:

        BM25
          +
        exact phrase
          +
        n-gram
          +
        category
          +
        rating prior
          ↓
        Reciprocal Rank Fusion
          ↓
        Maximal Marginal Relevance
          ↓
        Top-K

    V5 adds ONLY the MMR slate-diversification stage
    to the V4 architecture.

    The conversational state and retrieval routes remain
    intentionally unchanged so that the effect of MMR can
    be measured independently.
    """

    ROUTE_LIMIT = 100

    RRF_K = 60.0

    RRF_WEIGHTS = {
        "bm25": 1.00,
        "phrase": 1.20,
        "ngram": 0.85,
        "category": 0.90,
        "rating": 0.15,
    }

    BM25_WEIGHTS = (
        0.0,   # parent_asin
        6.0,   # title
        4.0,   # categories
        2.5,   # features
        2.5,   # details
        1.5,   # store
        1.0,   # description
    )

    RATING_PRIOR_STRENGTH = 20.0

    # ------------------------------------------------------------
    # V5 MMR parameter.
    #
    # High lambda means relevance remains dominant.
    #
    # 0.85 relevance
    # 0.15 diversity
    # ------------------------------------------------------------

    MMR_LAMBDA = 0.85

    SLOT_WEIGHTS = {
        "category": 3.0,
        "material": 2.5,
        "color": 2.5,
        "size": 2.5,
        "style": 2.0,
        "brand": 2.0,
        "budget": 2.5,
        "feature": 2.0,
        "use_case": 2.5,
        "other": 1.0,
    }

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> None:

        self.catalog_path = Path(catalog_path)

        self.connection = sqlite3.connect(":memory:")

        self._sessions: dict[str, dict] = {}

        self._products: dict[str, dict] = {}

        self._term_index: dict[str, set[str]] = defaultdict(set)
        self._phrase_index: dict[str, set[str]] = defaultdict(set)
        self._ngram_index: dict[str, set[str]] = defaultdict(set)
        self._category_index: dict[str, set[str]] = defaultdict(set)

        self._product_terms: dict[str, set[str]] = {}
        self._product_categories: dict[str, set[str]] = {}

        self._rating_prior: dict[str, float] = {}

        self._global_rating = 0.0

        self._build_index()

    # ============================================================
    # INDEX
    # ============================================================

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
        ratings = []

        with self.catalog_path.open(
            encoding="utf-8"
        ) as handle:

            for line in handle:

                product = json.loads(line)

                asin = str(
                    product["parent_asin"]
                )

                self._products[asin] = product

                title = _text(
                    product.get("title")
                )

                categories = _text(
                    product.get("categories")
                )

                features = _text(
                    product.get("features")
                )

                details = _text(
                    product.get("details")
                )

                store = _text(
                    product.get("store")
                )

                description = _text(
                    product.get("description")
                )

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

                searchable = " ".join(
                    [
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
                    ]
                )

                terms = _terms(searchable)

                term_set = set(terms)

                self._product_terms[
                    asin
                ] = term_set

                for term in term_set:
                    self._term_index[
                        term
                    ].add(asin)

                phrase_terms = _terms(
                    " ".join(
                        [
                            title,
                            categories,
                            features,
                            details,
                        ]
                    )
                )

                for phrase in _ngrams(
                    phrase_terms,
                    2,
                    3,
                ):
                    self._phrase_index[
                        phrase
                    ].add(asin)

                for ngram in _ngrams(
                    terms,
                    2,
                    3,
                ):
                    self._ngram_index[
                        ngram
                    ].add(asin)

                category_terms = set(
                    _terms(categories)
                )

                self._product_categories[
                    asin
                ] = category_terms

                for term in category_terms:
                    self._category_index[
                        term
                    ].add(asin)

                rating = self._get_rating(
                    product
                )

                if rating is not None:
                    ratings.append(rating)

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

        if ratings:
            self._global_rating = (
                sum(ratings) / len(ratings)
            )

        # Empirical Bayes rating shrinkage.
        for asin, product in self._products.items():

            rating = self._get_rating(
                product
            )

            count = self._get_rating_count(
                product
            )

            if rating is None:

                self._rating_prior[
                    asin
                ] = self._global_rating

                continue

            if count is None:
                count = 0.0

            self._rating_prior[
                asin
            ] = (
                count * rating
                +
                self.RATING_PRIOR_STRENGTH
                * self._global_rating
            ) / (
                count
                + self.RATING_PRIOR_STRENGTH
            )

    # ============================================================
    # PRODUCT METADATA
    # ============================================================

    @staticmethod
    def _get_rating(
        product: dict,
    ) -> float | None:

        for field in (
            "rating",
            "average_rating",
            "stars",
        ):

            value = product.get(field)

            if value in (None, ""):
                continue

            try:

                value = float(value)

                if 0.0 <= value <= 5.0:
                    return value

            except (
                TypeError,
                ValueError,
            ):
                pass

        return None

    @staticmethod
    def _get_rating_count(
        product: dict,
    ) -> float | None:

        for field in (
            "rating_count",
            "ratings_count",
            "review_count",
            "reviews_count",
            "num_reviews",
        ):

            value = product.get(field)

            if value in (None, ""):
                continue

            try:

                value = float(value)

                if value >= 0:
                    return value

            except (
                TypeError,
                ValueError,
            ):
                pass

        return None

    # ============================================================
    # SESSION
    # ============================================================

    def reset(
        self,
        session_id: str,
        user_profile: dict,
    ) -> None:

        self._sessions[
            session_id
        ] = {
            "user_profile": (
                user_profile or {}
            ),
            "history": [],
            "slots": defaultdict(list),
            "goal_terms": [],
            "context_terms": [],
            "overridden_terms": set(),
            "intent": "browsing",
            "override_count": 0,
            "last_message": "",
        }

    # ============================================================
    # INTENT
    # ============================================================

    def _is_override(
        self,
        message: str,
    ) -> bool:

        lowered = message.lower()

        return any(
            pattern in lowered
            for pattern in OVERRIDE_PATTERNS
        )

    def _detect_intent(
        self,
        message: str,
        state: dict,
    ) -> str:

        lowered = message.lower()

        if self._is_override(message):
            return "buying"

        buying_signals = (
            "need",
            "must",
            "require",
            "requirement",
            "looking for",
            "buy",
            "purchase",
            "want",
            "budget",
            "under",
            "$",
        )

        browsing_signals = (
            "exploring",
            "browse",
            "browsing",
            "options",
            "ideas",
            "not sure",
            "anything",
            "still exploring",
        )

        buying_score = sum(
            signal in lowered
            for signal in buying_signals
        )

        browsing_score = sum(
            signal in lowered
            for signal in browsing_signals
        )

        if buying_score > browsing_score:
            return "buying"

        if browsing_score > 0:
            return "browsing"

        if any(
            state["slots"].values()
        ):
            return "buying"

        return state.get(
            "intent",
            "browsing",
        )

    # ============================================================
    # SLOT EXTRACTION
    # ============================================================

    def _extract_budget(
        self,
        message: str,
    ) -> list[str]:

        results = []

        patterns = (
            r"(?:under|below|less than|up to)"
            r"\s*\$?\s*(\d+(?:\.\d+)?)",

            r"\$\s*(\d+(?:\.\d+)?)",

            r"(\d+(?:\.\d+)?)\s*dollars",
        )

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                message.lower(),
            ):

                results.append(
                    f"budget {match.group(1)}"
                )

        return list(
            dict.fromkeys(results)
        )

    def _extract_slots(
        self,
        message: str,
    ) -> dict[str, list[str]]:

        lowered = message.lower()

        slots = defaultdict(list)

        for material in MATERIALS:

            if re.search(
                rf"\b{re.escape(material)}\b",
                lowered,
            ):

                slots[
                    "material"
                ].append(material)

        for color in COLORS:

            if re.search(
                rf"\b{re.escape(color)}\b",
                lowered,
            ):

                slots[
                    "color"
                ].append(color)

        for use_case in USE_CASES:

            if re.search(
                rf"\b{re.escape(use_case)}\b",
                lowered,
            ):

                slots[
                    "use_case"
                ].append(use_case)

        for value in self._extract_budget(
            message
        ):

            slots[
                "budget"
            ].append(value)

        size_patterns = (
            r"\bsize\s+([a-z0-9]+)",
            r"\bsizing\s+([a-z0-9]+)",
        )

        for pattern in size_patterns:

            for match in re.finditer(
                pattern,
                lowered,
            ):

                slots[
                    "size"
                ].append(
                    match.group(1)
                )

        features = (
            "waterproof",
            "water resistant",
            "lightweight",
            "durable",
            "casual",
            "formal",
            "comfortable",
            "slim fit",
        )

        for feature in features:

            if feature in lowered:

                slots[
                    "feature"
                ].append(feature)

        return {
            key: list(
                dict.fromkeys(values)
            )
            for key, values in slots.items()
        }

    # ============================================================
    # STATE UPDATE
    # ============================================================

    def _update_state(
        self,
        state: dict,
        message: str,
    ) -> None:

        override = self._is_override(
            message
        )

        extracted = self._extract_slots(
            message
        )

        current_terms = _terms(
            message
        )

        if not override:

            for attribute, values in extracted.items():

                for value in values:

                    if value not in state[
                        "slots"
                    ][attribute]:

                        state[
                            "slots"
                        ][attribute].append(
                            value
                        )

            state[
                "goal_terms"
            ] = list(
                dict.fromkeys(
                    [
                        *state["goal_terms"],
                        *current_terms,
                    ]
                )
            )[-80:]

            state[
                "context_terms"
            ] = list(
                dict.fromkeys(
                    [
                        *state["context_terms"],
                        *current_terms,
                    ]
                )
            )[-80:]

        else:

            state[
                "override_count"
            ] += 1

            new_values = set()

            for values in extracted.values():
                new_values.update(values)

            # Replace only the slots explicitly changed
            # by the new message.
            for attribute, values in extracted.items():

                if values:

                    state[
                        "slots"
                    ][attribute] = list(
                        dict.fromkeys(
                            values
                        )
                    )

            old_terms = set(
                state["goal_terms"]
            )

            explicit_new_terms = set(
                current_terms
            )

            for old_term in old_terms:

                if (
                    old_term not in explicit_new_terms
                    and old_term in {
                        "cotton",
                        "polyester",
                        "nylon",
                        "leather",
                        "wool",
                        "spandex",
                        "silk",
                        "rayon",
                        "fabric",
                        *COLORS,
                        *USE_CASES,
                    }
                ):

                    if new_values:

                        state[
                            "overridden_terms"
                        ].add(
                            old_term
                        )

            replacement_terms = [
                term
                for term in current_terms
                if term not in {
                    "actually",
                    "ignore",
                    "earlier",
                    "previous",
                    "preference",
                    "changed",
                    "mind",
                }
            ]

            active_goal = [
                term
                for term in state[
                    "goal_terms"
                ]
                if term not in state[
                    "overridden_terms"
                ]
            ]

            state[
                "goal_terms"
            ] = list(
                dict.fromkeys(
                    [
                        *active_goal,
                        *replacement_terms,
                    ]
                )
            )[-80:]

            state[
                "context_terms"
            ] = list(
                dict.fromkeys(
                    [
                        *state["context_terms"],
                        *current_terms,
                    ]
                )
            )[-80:]

        state[
            "intent"
        ] = self._detect_intent(
            message,
            state,
        )

    # ============================================================
    # QUERY BUILDER
    # ============================================================

    def _build_query(
        self,
        state: dict,
    ) -> str:

        query_terms = []

        query_terms.extend(
            state.get(
                "goal_terms",
                [],
            )
        )

        for attribute, values in state[
            "slots"
        ].items():

            weight = self.SLOT_WEIGHTS.get(
                attribute,
                1.0,
            )

            repeat = max(
                1,
                min(
                    3,
                    int(
                        math.ceil(weight)
                    ),
                ),
            )

            for value in values:

                query_terms.extend(
                    [value] * repeat
                )

        active_terms = set(
            query_terms
        )

        context_added = 0

        for term in reversed(
            state.get(
                "context_terms",
                [],
            )
        ):

            if term in active_terms:
                continue

            if term in state[
                "overridden_terms"
            ]:
                continue

            query_terms.append(term)

            context_added += 1

            if context_added >= 30:
                break

        profile = state.get(
            "user_profile",
            {},
        )

        if isinstance(profile, dict):

            for key, value in profile.items():

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                    ),
                ):

                    query_terms.append(
                        str(value)
                    )

                elif isinstance(
                    value,
                    list,
                ):

                    query_terms.extend(
                        str(item)
                        for item in value[:5]
                    )

        return " ".join(
            query_terms
        ).strip()

    # ============================================================
    # BM25
    # ============================================================

    def _bm25(
        self,
        query: str,
        limit: int,
    ) -> list[str]:

        terms = list(
            dict.fromkeys(
                _terms(query)
            )
        )[:80]

        if not terms:
            return []

        expression = " OR ".join(
            f'"{term}"'
            for term in terms
        )

        weights = ", ".join(
            str(value)
            for value in self.BM25_WEIGHTS
        )

        rows = self.connection.execute(
            "SELECT parent_asin "
            "FROM products "
            "WHERE products MATCH ? "
            f"ORDER BY bm25(products, {weights}) "
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

    # ============================================================
    # PHRASE ROUTE
    # ============================================================

    def _phrase(
        self,
        query: str,
        limit: int,
    ) -> list[str]:

        terms = _terms(query)

        if len(terms) < 2:
            return []

        phrases = _ngrams(
            terms,
            2,
            3,
        )

        scores = Counter()

        for phrase in phrases:

            length = len(
                phrase.split()
            )

            score = float(
                length * length
            )

            for asin in self._phrase_index.get(
                phrase,
                set(),
            ):

                scores[
                    asin
                ] += score

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            asin
            for asin, _ in ranked[:limit]
        ]

    # ============================================================
    # N-GRAM ROUTE
    # ============================================================

    def _ngram(
        self,
        query: str,
        limit: int,
    ) -> list[str]:

        terms = _terms(query)

        if len(terms) < 2:
            return []

        grams = _ngrams(
            terms,
            2,
            3,
        )

        scores = Counter()

        for gram in grams:

            for asin in self._ngram_index.get(
                gram,
                set(),
            ):

                scores[
                    asin
                ] += 1

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            asin
            for asin, _ in ranked[:limit]
        ]

    # ============================================================
    # CATEGORY ROUTE
    # ============================================================

    def _category(
        self,
        query: str,
        limit: int,
    ) -> list[str]:

        terms = set(
            _terms(query)
        )

        scores = Counter()

        for term in terms:

            for asin in self._category_index.get(
                term,
                set(),
            ):

                scores[
                    asin
                ] += 1

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            asin
            for asin, _ in ranked[:limit]
        ]

    # ============================================================
    # RATING ROUTE
    # ============================================================

    def _rating(
        self,
        query: str,
        candidates: set[str],
        limit: int,
    ) -> list[str]:

        terms = set(
            _terms(query)
        )

        if not terms:
            return []

        scored = []

        for asin in candidates:

            product_terms = (
                self._product_terms.get(
                    asin,
                    set(),
                )
            )

            overlap = (
                terms
                &
                product_terms
            )

            if not overlap:
                continue

            relevance = (
                len(overlap)
                /
                max(
                    1,
                    len(terms),
                )
            )

            rating = self._rating_prior.get(
                asin,
                self._global_rating,
            )

            scored.append(
                (
                    asin,
                    relevance * rating,
                )
            )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            asin
            for asin, _ in scored[:limit]
        ]

    # ============================================================
    # RRF
    # ============================================================

    def _rrf(
        self,
        routes: dict[str, list[str]],
        limit: int,
    ) -> list[str]:

        scores = defaultdict(float)

        for route, results in routes.items():

            weight = self.RRF_WEIGHTS.get(
                route,
                1.0,
            )

            for rank, asin in enumerate(
                results,
                start=1,
            ):

                scores[
                    asin
                ] += (
                    weight
                    /
                    (
                        self.RRF_K
                        + rank
                    )
                )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            asin
            for asin, _ in ranked[:limit]
        ]

    # ============================================================
    # V5: PRODUCT SIMILARITY
    # ============================================================

    def _product_similarity(
        self,
        asin_a: str,
        asin_b: str,
    ) -> float:

        terms_a = self._product_terms.get(
            asin_a,
            set(),
        )

        terms_b = self._product_terms.get(
            asin_b,
            set(),
        )

        if not terms_a or not terms_b:
            return 0.0

        intersection = len(
            terms_a & terms_b
        )

        union = len(
            terms_a | terms_b
        )

        if union == 0:
            return 0.0

        return intersection / union

    # ============================================================
    # V5: MMR
    # ============================================================

    def _mmr(
        self,
        candidates: list[str],
        rrf_scores: dict[str, float],
        limit: int,
    ) -> list[str]:

        if not candidates:
            return []

        if len(candidates) <= limit:
            return candidates[:limit]

        # --------------------------------------------------------
        # Normalize RRF scores into [0, 1].
        # --------------------------------------------------------

        values = [
            rrf_scores.get(
                asin,
                0.0,
            )
            for asin in candidates
        ]

        maximum = max(
            values,
            default=0.0,
        )

        minimum = min(
            values,
            default=0.0,
        )

        score_range = (
            maximum - minimum
        )

        if score_range <= 0:
            normalized = {
                asin: 1.0
                for asin in candidates
            }

        else:

            normalized = {
                asin: (
                    rrf_scores.get(
                        asin,
                        0.0,
                    )
                    - minimum
                ) / score_range
                for asin in candidates
            }

        # --------------------------------------------------------
        # Greedy MMR.
        # --------------------------------------------------------

        remaining = list(
            candidates
        )

        selected: list[str] = []

        while (
            remaining
            and len(selected) < limit
        ):

            best_asin = None
            best_score = float("-inf")

            for asin in remaining:

                relevance = normalized.get(
                    asin,
                    0.0,
                )

                if not selected:

                    redundancy = 0.0

                else:

                    redundancy = max(
                        (
                            self._product_similarity(
                                asin,
                                selected_asin,
                            )
                            for selected_asin in selected
                        ),
                        default=0.0,
                    )

                mmr_score = (
                    self.MMR_LAMBDA
                    * relevance
                    -
                    (
                        1.0
                        -
                        self.MMR_LAMBDA
                    )
                    * redundancy
                )

                # Deterministic tie-breaking:
                # retain earlier RRF ranking.
                if (
                    mmr_score > best_score
                ):

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

    # ============================================================
    # ASK ATTRIBUTE
    # ============================================================

    def _ask_attribute(
        self,
        state: dict,
    ) -> str:

        if state.get(
            "intent"
        ) == "buying":

            order = [
                "material",
                "color",
                "size",
                "budget",
                "style",
                "feature",
                "use_case",
                "brand",
            ]

        else:

            order = [
                "use_case",
                "style",
                "material",
                "color",
                "feature",
                "budget",
            ]

        for attribute in order:

            if not state[
                "slots"
            ].get(attribute):

                return attribute

        return "other"

    # ============================================================
    # RESPOND
    # ============================================================

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

        state[
            "history"
        ].append(
            user_message
        )

        state[
            "last_message"
        ] = user_message

        # --------------------------------------------------------
        # Update conversation state.
        # --------------------------------------------------------

        self._update_state(
            state,
            user_message,
        )

        # --------------------------------------------------------
        # Build active retrieval query.
        # --------------------------------------------------------

        query = self._build_query(
            state
        )

        # --------------------------------------------------------
        # Independent retrieval routes.
        # --------------------------------------------------------

        bm25_results = self._bm25(
            query,
            self.ROUTE_LIMIT,
        )

        phrase_results = self._phrase(
            query,
            self.ROUTE_LIMIT,
        )

        ngram_results = self._ngram(
            query,
            self.ROUTE_LIMIT,
        )

        category_results = self._category(
            query,
            self.ROUTE_LIMIT,
        )

        candidate_pool = set(
            bm25_results
        )

        candidate_pool.update(
            phrase_results
        )

        candidate_pool.update(
            ngram_results
        )

        candidate_pool.update(
            category_results
        )

        rating_results = self._rating(
            query,
            candidate_pool,
            self.ROUTE_LIMIT,
        )

        # --------------------------------------------------------
        # RRF with explicit scores.
        # --------------------------------------------------------

        routes = {
            "bm25": bm25_results,
            "phrase": phrase_results,
            "ngram": ngram_results,
            "category": category_results,
            "rating": rating_results,
        }

        rrf_scores = defaultdict(float)

        for route, results in routes.items():

            weight = self.RRF_WEIGHTS.get(
                route,
                1.0,
            )

            for rank, asin in enumerate(
                results,
                start=1,
            ):

                rrf_scores[
                    asin
                ] += (
                    weight
                    /
                    (
                        self.RRF_K
                        + rank
                    )
                )

        rrf_ranked = [
            asin
            for asin, _ in sorted(
                rrf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        # --------------------------------------------------------
        # V5 MMR.
        #
        # Use a reasonably large RRF candidate pool so that MMR
        # can improve diversity without restricting itself to
        # the first 10 RRF results.
        # --------------------------------------------------------

        mmr_candidates = rrf_ranked[
            : max(
                50,
                top_k * 5,
            )
        ]

        final_ranked = self._mmr(
            mmr_candidates,
            dict(rrf_scores),
            top_k,
        )

        recommendations = [
            {
                "parent_asin": asin
            }
            for asin in final_ranked
        ]

        # --------------------------------------------------------
        # Conversational guidance.
        # --------------------------------------------------------

        ask_attribute = None

        if turn < 10:

            ask_attribute = (
                self._ask_attribute(
                    state
                )
            )

        if state[
            "override_count"
        ] > 0:

            message = (
                "Understood. I have updated "
                "the search based on your latest "
                "requirements."
            )

        elif state[
            "intent"
        ] == "browsing":

            message = (
                "Here are some options to "
                "explore. I can narrow them "
                "down further if needed."
            )

        else:

            message = (
                "Here are the closest matches "
                "to your current requirements."
            )

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }

