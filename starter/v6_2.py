from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "something", "need", "like", "prefer",
    "find", "show", "give", "get", "can", "could", "do", "have",
    "has", "had", "about", "also", "just", "really", "very"
}


BUYING_WORDS = {
    "buy", "purchase", "need", "want", "require", "required",
    "must", "under", "below", "budget", "size", "color", "colour",
    "material", "price", "cheap", "affordable", "work", "running",
    "hiking", "gym", "winter", "school", "office", "formal",
    "casual", "men", "women", "woman", "man", "kids", "children"
}


BROWSING_WORDS = {
    "exploring", "explore", "browse", "browsing", "ideas", "options",
    "suggest", "suggestions", "recommend", "recommendations", "similar",
    "interesting", "maybe", "perhaps", "discover", "inspiration",
    "different", "types", "style", "styles", "choices"
}


ATTRIBUTE_WORDS = {
    "black", "white", "blue", "red", "green", "brown", "pink",
    "purple", "yellow", "orange", "grey", "gray",

    "cotton", "leather", "wool", "nylon", "polyester", "denim",
    "suede", "canvas", "linen", "silk",

    "small", "medium", "large", "xl", "xxl",

    "running", "hiking", "gym", "training", "walking", "tennis",
    "basketball", "soccer", "football",

    "winter", "summer", "rain", "waterproof",

    "men", "women", "woman", "man", "boys", "girls", "kids",

    "formal", "casual", "work", "office", "wedding", "party"
}


PRODUCT_TYPE_WORDS = {
    "shoes", "shoe", "boots", "boot", "sneakers", "sneaker",
    "sandals", "sandal", "heels", "heel", "flats",
    "shirt", "shirts", "tshirt", "tshirts", "top", "tops",
    "pants", "trousers", "jeans", "shorts", "skirt", "skirts",
    "dress", "dresses", "jacket", "jackets", "coat", "coats",
    "sweater", "sweaters", "hoodie", "hoodies",
    "bag", "bags", "backpack", "backpacks", "purse", "purses",
    "hat", "hats", "cap", "caps", "belt", "belts",
    "watch", "watches", "jewelry", "necklace", "bracelet",
    "gloves", "socks", "sock", "underwear"
}


OVERRIDE_WORDS = {
    "actually",
    "instead",
    "rather",
    "change",
    "changed",
    "switch",
    "switched",
    "never",
    "mind",
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
        return " ".join(
            str(item)
            for item in value
        )

    return str(value)


def _terms(text: str) -> list[str]:

    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if (
            len(token) > 1
            and token.lower() not in STOPWORDS
        )
    ]


def _ngrams(
    tokens: list[str],
    n: int,
) -> list[str]:

    if len(tokens) < n:
        return []

    return [
        " ".join(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


class Agent:
    """
    V6.3

    Hybrid conversational shopping agent.

    V6.3 combines the strongest behaviour observed in V6.1
    and V6.2.

    Main components:

    - Buying / Browsing intent routing
    - Explicit conversation state
    - Current-turn priority
    - Override-aware context handling
    - Weighted BM25
    - Phrase retrieval
    - Category retrieval
    - RRF
    - Intent-specific reranking
    - Conservative MMR for browsing
    - Relevance-first ranking for buying
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> None:

        self.catalog_path = Path(
            catalog_path
        )

        self.connection = sqlite3.connect(
            ":memory:"
        )

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

                asin = str(
                    product["parent_asin"]
                )

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

                rating = product.get(
                    "average_rating"
                )

                try:
                    rating = float(rating)
                except (
                    TypeError,
                    ValueError,
                ):
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

            # Explicit conversational state.
            "persistent_terms": set(),
            "active_attributes": set(),
            "active_products": set(),
            "overridden_terms": set(),
        }

    # =========================================================
    # INTENT
    # =========================================================

    def _detect_intent(
        self,
        current_message: str,
        history: list[str],
    ) -> str:

        current_tokens = set(
            _terms(current_message)
        )

        buying_score = len(
            current_tokens
            & BUYING_WORDS
        )

        browsing_score = len(
            current_tokens
            & BROWSING_WORDS
        )

        if buying_score > browsing_score:
            return "buying"

        if browsing_score > buying_score:
            return "browsing"

        if current_tokens & ATTRIBUTE_WORDS:
            return "buying"

        if current_tokens & PRODUCT_TYPE_WORDS:
            return "buying"

        recent = history[-3:]

        recent_tokens = set(
            _terms(
                " ".join(recent)
            )
        )

        buying_score = len(
            recent_tokens
            & BUYING_WORDS
        )

        browsing_score = len(
            recent_tokens
            & BROWSING_WORDS
        )

        if buying_score > browsing_score:
            return "buying"

        return "browsing"

    # =========================================================
    # ATTRIBUTE EXTRACTION
    # =========================================================

    def _extract_attributes(
        self,
        text: str,
    ) -> set[str]:

        tokens = set(
            _terms(text)
        )

        return (
            tokens
            & ATTRIBUTE_WORDS
        )

    # =========================================================
    # PRODUCT EXTRACTION
    # =========================================================

    def _extract_products(
        self,
        text: str,
    ) -> set[str]:

        tokens = set(
            _terms(text)
        )

        return (
            tokens
            & PRODUCT_TYPE_WORDS
        )

    # =========================================================
    # OVERRIDE DETECTION
    # =========================================================

    def _is_override(
        self,
        text: str,
    ) -> bool:

        lower = text.lower()

        return (
            "ignore my earlier" in lower
            or "ignore earlier" in lower
            or "actually" in lower
            or "instead" in lower
            or "change that" in lower
            or "never mind" in lower
            or "rather" in lower
        )

    # =========================================================
    # STATE UPDATE
    # =========================================================

    def _update_state(
        self,
        state: dict,
        message: str,
        override: bool,
    ) -> None:

        tokens = set(
            _terms(message)
        )

        attributes = (
            tokens
            & ATTRIBUTE_WORDS
        )

        products = (
            tokens
            & PRODUCT_TYPE_WORDS
        )

        # -----------------------------------------------------
        # Override behaviour
        #
        # New attributes become active.
        # Old conflicting attributes are removed.
        # -----------------------------------------------------

        if override:

            old_attributes = state[
                "active_attributes"
            ]

            # A new attribute supersedes old attributes
            # when they belong to the same broad class.
            #
            # Colour replacement.
            colours = {
                "black", "white", "blue", "red",
                "green", "brown", "pink", "purple",
                "yellow", "orange", "grey", "gray"
            }

            materials = {
                "cotton", "leather", "wool", "nylon",
                "polyester", "denim", "suede",
                "canvas", "linen", "silk"
            }

            activities = {
                "running", "hiking", "gym", "training",
                "walking", "tennis", "basketball",
                "soccer", "football"
            }

            for group in (
                colours,
                materials,
                activities,
            ):

                new_group = (
                    attributes & group
                )

                if new_group:

                    for old in list(
                        old_attributes
                    ):

                        if old in group:
                            state[
                                "overridden_terms"
                            ].add(old)

                            old_attributes.remove(
                                old
                            )

                    old_attributes.update(
                        new_group
                    )

            state[
                "active_products"
            ].update(products)

        else:

            state[
                "active_attributes"
            ].update(attributes)

            state[
                "active_products"
            ].update(products)

        # -----------------------------------------------------
        # Persistent context
        #
        # Keep useful terms that are not explicit attributes.
        # -----------------------------------------------------

        persistent = (
            tokens
            - STOPWORDS
            - OVERRIDE_WORDS
        )

        state[
            "persistent_terms"
        ].update(persistent)

    # =========================================================
    # QUERY BUILDING
    # =========================================================

    def _build_query(
        self,
        state: dict,
        current_message: str,
        override: bool,
    ) -> tuple[str, list[str]]:

        current_tokens = _terms(
            current_message
        )

        current_set = set(
            current_tokens
        )

        # -----------------------------------------------------
        # Persistent product/category context.
        # -----------------------------------------------------

        persistent = list(
            state[
                "persistent_terms"
            ]
        )

        # Remove explicitly overridden terms.
        persistent = [
            token
            for token in persistent
            if token not in state[
                "overridden_terms"
            ]
        ]

        # -----------------------------------------------------
        # Explicit active slots.
        # -----------------------------------------------------

        active_attributes = list(
            state[
                "active_attributes"
            ]
        )

        active_products = list(
            state[
                "active_products"
            ]
        )

        # -----------------------------------------------------
        # Build priority ordering.
        #
        # Current turn > active slots > persistent context.
        # -----------------------------------------------------

        ordered = []

        # Current message gets strongest influence.
        ordered.extend(
            current_tokens
        )

        # Active product types.
        ordered.extend(
            active_products
        )

        # Active attributes.
        ordered.extend(
            active_attributes
        )

        # Older persistent context.
        ordered.extend(
            persistent
        )

        # Remove duplicates while preserving priority.
        tokens = list(
            dict.fromkeys(
                ordered
            )
        )

        # Current terms should always survive.
        for token in current_set:

            if token not in tokens:
                tokens.insert(
                    0,
                    token
                )

        # Keep query compact.
        tokens = tokens[:100]

        if not tokens:
            return "", []

        expression = " OR ".join(
            f'"{term}"'
            for term in tokens
        )

        return (
            expression,
            tokens,
        )

    # =========================================================
    # BM25
    # =========================================================

    def _bm25_candidates(
        self,
        expression: str,
        limit: int = 100,
        intent: str = "buying",
    ) -> list[str]:

        if not expression:
            return []

        if intent == "buying":

            weights = (
                8.0,
                3.0,
                6.0,
                3.0,
                1.5,
                2.0,
            )

        else:

            weights = (
                5.0,
                6.0,
                5.0,
                3.0,
                1.5,
                4.0,
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
    # PHRASE RETRIEVAL
    # =========================================================

    def _phrase_candidates(
        self,
        tokens: list[str],
        limit: int = 100,
    ) -> list[str]:

        phrases = []

        phrases.extend(
            _ngrams(tokens, 3)
        )

        phrases.extend(
            _ngrams(tokens, 2)
        )

        phrases = phrases[-30:]

        scores: Counter[str] = Counter()

        for phrase in phrases:

            expression = (
                '"'
                + phrase.replace(
                    '"',
                    ""
                )
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

                asin = str(
                    row[0]
                )

                scores[asin] += (
                    1.0
                    / (
                        20.0
                        + rank
                    )
                )

        return [
            asin
            for asin, _ in scores.most_common(
                limit
            )
        ]

    # =========================================================
    # CATEGORY
    # =========================================================

    def _category_candidates(
        self,
        tokens: list[str],
        limit: int = 100,
    ) -> list[str]:

        if not tokens:
            return []

        category_terms = [
            token
            for token in tokens
            if (
                token in PRODUCT_TYPE_WORDS
                or token in ATTRIBUTE_WORDS
            )
        ]

        if not category_terms:
            category_terms = tokens[-25:]

        category_terms = list(
            dict.fromkeys(
                category_terms
            )
        )

        expression = " OR ".join(
            f'"{term}"'
            for term in category_terms
        )

        try:

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

        except sqlite3.Error:
            return []

        return [
            str(row[0])
            for row in rows
        ]

    # =========================================================
    # RATING
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
    # RRF
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
                    scores.get(
                        asin,
                        0.0
                    )
                    + weight
                    / (
                        k + rank
                    )
                )

        return scores

    # =========================================================
    # BUYING RERANK
    # =========================================================

    def _rerank_buying(
        self,
        candidates: list[str],
        tokens: list[str],
        top_k: int,
    ) -> list[str]:

        query_text = " ".join(
            tokens
        )

        query_set = set(
            tokens
        )

        query_2grams = set(
            _ngrams(tokens, 2)
        )

        query_3grams = set(
            _ngrams(tokens, 3)
        )

        attributes = (
            query_set
            & ATTRIBUTE_WORDS
        )

        products = (
            query_set
            & PRODUCT_TYPE_WORDS
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

            title = metadata[
                "title"
            ]

            features = metadata[
                "features"
            ]

            categories = metadata[
                "categories"
            ]

            combined = (
                title
                + " "
                + features
                + " "
                + categories
            )

            title_terms = set(
                _terms(title)
            )

            combined_terms = set(
                _terms(combined)
            )

            score = 0.0

            # -------------------------------------------------
            # General coverage
            # -------------------------------------------------

            if query_set:

                coverage = (
                    len(
                        query_set
                        & combined_terms
                    )
                    / len(query_set)
                )

                score += (
                    0.10
                    * coverage
                )

            # -------------------------------------------------
            # Title coverage
            #
            # Restored closer to V6.1 behaviour.
            # -------------------------------------------------

            if query_set:

                title_coverage = (
                    len(
                        query_set
                        & title_terms
                    )
                    / len(query_set)
                )

                score += (
                    0.20
                    * title_coverage
                )

            # -------------------------------------------------
            # Product type
            # -------------------------------------------------

            if products:

                matched = len(
                    products
                    & (
                        title_terms
                        | set(
                            _terms(
                                categories
                            )
                        )
                    )
                )

                score += (
                    0.22
                    * (
                        matched
                        / len(products)
                    )
                )

            # -------------------------------------------------
            # Attributes
            # -------------------------------------------------

            if attributes:

                matched = len(
                    attributes
                    & combined_terms
                )

                score += (
                    0.18
                    * (
                        matched
                        / len(attributes)
                    )
                )

            # -------------------------------------------------
            # Exact full phrase
            # -------------------------------------------------

            if (
                query_text
                and query_text in title
            ):

                score += 0.25

            elif (
                query_text
                and query_text in combined
            ):

                score += 0.10

            # -------------------------------------------------
            # 3-grams
            # -------------------------------------------------

            for phrase in query_3grams:

                if phrase in title:
                    score += 0.060

                elif phrase in combined:
                    score += 0.030

            # -------------------------------------------------
            # 2-grams
            # -------------------------------------------------

            for phrase in query_2grams:

                if phrase in title:
                    score += 0.040

                elif phrase in combined:
                    score += 0.020

            # -------------------------------------------------
            # Rating remains very small.
            # -------------------------------------------------

            score += (
                0.010
                * self._rating_score(
                    asin
                )
            )

            # Preserve BM25/RRF ordering.
            score += (
                1.0
                / (
                    1000.0
                    + base_rank
                )
            )

            scored.append(
                (
                    score,
                    asin,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            asin
            for _, asin in scored[
                :max(
                    top_k * 5,
                    40
                )
            ]
        ]

    # =========================================================
    # BROWSING RERANK
    # =========================================================

    def _rerank_browsing(
        self,
        candidates: list[str],
        tokens: list[str],
        top_k: int,
    ) -> list[str]:

        query_set = set(
            tokens
        )

        products = (
            query_set
            & PRODUCT_TYPE_WORDS
        )

        attributes = (
            query_set
            & ATTRIBUTE_WORDS
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

            title = metadata[
                "title"
            ]

            categories = metadata[
                "categories"
            ]

            features = metadata[
                "features"
            ]

            title_terms = set(
                _terms(title)
            )

            category_terms = set(
                _terms(categories)
            )

            feature_terms = set(
                _terms(features)
            )

            combined_terms = (
                title_terms
                | category_terms
                | feature_terms
            )

            score = 0.0

            if query_set:

                coverage = (
                    len(
                        query_set
                        & combined_terms
                    )
                    / len(query_set)
                )

                score += (
                    0.16
                    * coverage
                )

            # Category is particularly important for browsing.
            if query_set:

                category_coverage = (
                    len(
                        query_set
                        & category_terms
                    )
                    / len(query_set)
                )

                score += (
                    0.18
                    * category_coverage
                )

            # Title.
            if query_set:

                title_coverage = (
                    len(
                        query_set
                        & title_terms
                    )
                    / len(query_set)
                )

                score += (
                    0.18
                    * title_coverage
                )

            # Product type.
            if products:

                matched = len(
                    products
                    & (
                        title_terms
                        | category_terms
                    )
                )

                score += (
                    0.18
                    * (
                        matched
                        / len(products)
                    )
                )

            # Attributes.
            if attributes:

                matched = len(
                    attributes
                    & combined_terms
                )

                score += (
                    0.12
                    * (
                        matched
                        / len(attributes)
                    )
                )

            # Rating is only a tie-breaker.
            score += (
                0.006
                * self._rating_score(
                    asin
                )
            )

            # Preserve retrieval order.
            score += (
                1.0
                / (
                    3000.0
                    + base_rank
                )
            )

            scored.append(
                (
                    score,
                    asin,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            asin
            for _, asin in scored[
                :max(
                    top_k * 8,
                    60
                )
            ]
        ]

    # =========================================================
    # CONSERVATIVE BROWSING MMR
    # =========================================================

    def _mmr_browsing(
        self,
        candidates: list[str],
        tokens: list[str],
        top_k: int,
    ) -> list[str]:

        if not candidates:
            return []

        # Preserve the strongest candidate.
        selected = [
            candidates[0]
        ]

        if top_k == 1:
            return selected

        query_set = set(
            tokens
        )

        candidate_sets = {}

        for asin in candidates:

            metadata = self._metadata.get(
                asin,
                {}
            )

            text = (
                metadata.get(
                    "title",
                    ""
                )
                + " "
                + metadata.get(
                    "features",
                    ""
                )
                + " "
                + metadata.get(
                    "categories",
                    ""
                )
            )

            candidate_sets[asin] = set(
                _terms(text)
            )

        remaining = [
            asin
            for asin in candidates
            if asin != candidates[0]
        ]

        while (
            remaining
            and len(selected) < top_k
        ):

            best = None
            best_score = -float(
                "inf"
            )

            for asin in remaining:

                candidate_set = (
                    candidate_sets[
                        asin
                    ]
                )

                if query_set:

                    relevance = (
                        len(
                            query_set
                            & candidate_set
                        )
                        / len(query_set)
                    )

                else:
                    relevance = 0.0

                similarity = 0.0

                for chosen in selected:

                    chosen_set = (
                        candidate_sets[
                            chosen
                        ]
                    )

                    union = len(
                        candidate_set
                        | chosen_set
                    )

                    if union:

                        similarity = max(
                            similarity,
                            len(
                                candidate_set
                                & chosen_set
                            )
                            / union
                        )

                # Very conservative diversity.
                score = (
                    0.94 * relevance
                    - 0.06 * similarity
                )

                rank = candidates.index(
                    asin
                )

                score += (
                    0.001
                    / (
                        1 + rank
                    )
                )

                if score > best_score:

                    best_score = score
                    best = asin

            if best is None:
                break

            selected.append(
                best
            )

            remaining.remove(
                best
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

        # -----------------------------------------------------
        # Detect override before updating state.
        # -----------------------------------------------------

        is_override = self._is_override(
            user_message
        )

        if is_override:

            state[
                "override_active"
            ] = True

        # -----------------------------------------------------
        # Add message to history.
        # -----------------------------------------------------

        state[
            "history"
        ].append(
            user_message
        )

        # -----------------------------------------------------
        # Update explicit state.
        # -----------------------------------------------------

        self._update_state(
            state,
            user_message,
            is_override,
        )

        # -----------------------------------------------------
        # Intent routing.
        # -----------------------------------------------------

        intent = self._detect_intent(
            user_message,
            state["history"],
        )

        state[
            "intent"
        ] = intent

        # -----------------------------------------------------
        # Build query.
        # -----------------------------------------------------

        expression, tokens = (
            self._build_query(
                state,
                user_message,
                is_override,
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
                    limit=120,
                    intent=intent,
                )
            )

            phrase_route = (
                self._phrase_candidates(
                    tokens,
                    limit=100,
                )
            )

            category_route = (
                self._category_candidates(
                    tokens,
                    limit=100,
                )
            )

            # -------------------------------------------------
            # Intent-specific route weighting
            # -------------------------------------------------

            if intent == "buying":

                # Restore the stronger V6.1 buying weighting.
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
                    0.44,
                    0.36,
                    0.20,
                ]

            # -------------------------------------------------
            # RRF
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
                )[:160]
            ]

            # -------------------------------------------------
            # Intent-specific reranking
            # -------------------------------------------------

            if intent == "buying":

                reranked = (
                    self._rerank_buying(
                        candidates,
                        tokens,
                        top_k,
                    )
                )

                # Do NOT apply MMR to buying.
                #
                # This protects MRR and preserves the strongest
                # exact product matches.
                recommendations = (
                    reranked[
                        :top_k
                    ]
                )

            else:

                reranked = (
                    self._rerank_browsing(
                        candidates,
                        tokens,
                        top_k,
                    )
                )

                recommendations = (
                    self._mmr_browsing(
                        reranked,
                        tokens,
                        top_k,
                    )
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
                for asin in recommendations[
                    :top_k
                ]
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }