from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from collections import Counter


# =============================================================
# TOKENIZATION
# =============================================================

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "for", "something", "need", "like",
    "prefer", "find", "show", "get", "give", "can", "could", "do",
    "have", "has", "had", "about", "am", "was", "were", "very",
}


BUYING_WORDS = {
    "buy", "purchase", "need", "require", "required",
    "must", "under", "below", "budget", "size", "color",
    "colour", "material", "price", "cheap", "affordable",
    "work", "running", "hiking", "gym", "winter",
    "daily", "use", "wear", "wearing",
}


BROWSING_WORDS = {
    "exploring", "explore", "browse", "browsing", "ideas",
    "options", "suggest", "suggestions", "recommend",
    "recommendations", "similar", "interesting", "maybe",
    "perhaps", "discover", "inspiration", "inspire",
}


# =============================================================
# COMMON SHOPPING ATTRIBUTES
# =============================================================

COLORS = {
    "black", "white", "blue", "red", "green", "brown",
    "grey", "gray", "pink", "purple", "yellow", "orange",
    "beige", "cream", "navy", "maroon", "burgundy",
    "khaki", "gold", "silver",
}


MATERIALS = {
    "cotton", "leather", "wool", "nylon", "polyester",
    "denim", "suede", "canvas", "silk", "linen",
    "cashmere", "fleece", "rubber", "mesh",
}


SIZES = {
    "xxs", "xs", "small", "s", "medium", "m",
    "large", "l", "xl", "xxl", "xxxl",
}


CATEGORY_TERMS = {
    "shirt", "shirts", "tshirt", "tshirts", "tee",
    "top", "tops", "dress", "dresses", "skirt", "skirts",
    "pants", "trousers", "jeans", "shorts", "jacket",
    "jackets", "coat", "coats", "hoodie", "hoodies",
    "sweater", "sweaters", "sweatshirt", "sweatshirts",
    "shoes", "shoe", "boots", "boot", "sneakers",
    "sandals", "heels", "flats", "loafers",
    "socks", "sock", "hat", "hats", "cap", "caps",
    "bag", "bags", "backpack", "wallet", "belt",
    "gloves", "scarf", "underwear", "bra",
}


ATTRIBUTE_WORDS = (
    COLORS
    | MATERIALS
    | SIZES
)


# =============================================================
# TEXT HELPERS
# =============================================================

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
        for i in range(
            len(tokens) - n + 1
        )
    ]


# =============================================================
# AGENT
# =============================================================

class Agent:
    """
    V10

    Conversational hybrid shopping retrieval.

    Main changes from V9:

    - Override detection reverted to trigger on discourse
      markers alone again (V9's category-switch requirement
      hurt the intent_override scenario it was meant to fix,
      because "actually, in white instead" — same category,
      new attribute — stopped clearing the old attribute).
    - Override now always clears colors/materials/sizes when
      triggered, but only clears categories when a new
      category term is actually present in the message. This
      keeps attribute-only overrides working without wiping
      category context on every "actually"/"instead".
    - Clarification trigger still DISABLED
      (ENABLE_CLARIFICATION = False).
    - MMR diversity cap for browsing kept as-is (protects top
      3 rerank-order candidates from being displaced).
    """

    ENABLE_CLARIFICATION = False

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

            "user_profile": (
                user_profile or {}
            ),

            "history": [],

            "intent": None,

            "override_active": False,

            "stable_terms": set(),

            "attributes": {
                "colors": set(),
                "materials": set(),
                "sizes": set(),
                "categories": set(),
            },
        }

    # =========================================================
    # ATTRIBUTE EXTRACTION
    # =========================================================

    def _extract_attributes(
        self,
        text: str,
    ) -> dict[str, set[str]]:

        tokens = set(
            _terms(text)
        )

        return {
            "colors": tokens & COLORS,
            "materials": tokens & MATERIALS,
            "sizes": tokens & SIZES,
            "categories": tokens & CATEGORY_TERMS,
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
            current_tokens & BUYING_WORDS
        )

        browsing_score = len(
            current_tokens & BROWSING_WORDS
        )

        if browsing_score > buying_score:
            return "browsing"

        if buying_score > browsing_score:
            return "buying"

        attributes = self._extract_attributes(
            current_message
        )

        if any(
            attributes.values()
        ):
            return "buying"

        if history:

            previous_tokens = set(
                _terms(
                    history[-1]
                )
            )

            previous_buying = len(
                previous_tokens
                & BUYING_WORDS
            )

            previous_browsing = len(
                previous_tokens
                & BROWSING_WORDS
            )

            if previous_buying > previous_browsing:
                return "buying"

        return "browsing"

    # =========================================================
    # UPDATE CONVERSATIONAL STATE
    # =========================================================

    def _update_state(
        self,
        state: dict,
        message: str,
        is_override: bool,
        override_category: bool,
    ) -> None:

        attributes = self._extract_attributes(
            message
        )

        tokens = _terms(message)

        # -----------------------------------------------------
        # Override:
        #
        # Always clear the fast-changing attribute slots.
        # Only clear category context when the message
        # actually introduces a new category term — otherwise
        # "actually, in white instead" would lose the product
        # type along with the color.
        # -----------------------------------------------------

        if is_override:

            state["attributes"]["colors"] = set()
            state["attributes"]["materials"] = set()
            state["attributes"]["sizes"] = set()

            if override_category:

                state["attributes"]["categories"] = set()
                state["stable_terms"] = set()

        # -----------------------------------------------------
        # Stable categories persist.
        # -----------------------------------------------------

        state[
            "stable_terms"
        ].update(
            token
            for token in tokens
            if token in CATEGORY_TERMS
        )

        # -----------------------------------------------------
        # Explicit attributes are updated.
        # -----------------------------------------------------

        for key in (
            "colors",
            "materials",
            "sizes",
        ):

            if attributes[key]:

                state[
                    "attributes"
                ][key] = set(
                    attributes[key]
                )

        if attributes["categories"]:

            state[
                "attributes"
            ]["categories"].update(
                attributes["categories"]
            )

    # =========================================================
    # QUERY CONSTRUCTION
    # =========================================================

    def _build_query(
        self,
        state: dict,
        current_message: str,
        is_override: bool,
    ) -> tuple[str, list[str]]:

        current_tokens = _terms(
            current_message
        )

        stable_terms = list(
            state["stable_terms"]
        )

        attribute_terms = []

        for key in (
            "colors",
            "materials",
            "sizes",
            "categories",
        ):

            attribute_terms.extend(
                state[
                    "attributes"
                ][key]
            )

        weighted_tokens = []

        weighted_tokens.extend(
            current_tokens * 5
        )

        weighted_tokens.extend(
            stable_terms * 2
        )

        weighted_tokens.extend(
            attribute_terms * 4
        )

        tokens = list(
            dict.fromkeys(
                weighted_tokens
            )
        )

        tokens = tokens[-100:]

        if not tokens:
            return "", []

        expression = " OR ".join(
            f'"{term}"'
            for term in tokens
        )

        scoring_tokens = list(
            dict.fromkeys(
                current_tokens
                + stable_terms
                + attribute_terms
            )
        )

        return (
            expression,
            scoring_tokens,
        )

    # =========================================================
    # BM25
    # =========================================================

    def _bm25_candidates(
        self,
        expression: str,
        limit: int,
        intent: str,
    ) -> list[str]:

        if not expression:
            return []

        if intent == "buying":

            weights = (
                11.0,
                3.0,
                7.0,
                2.0,
                1.0,
                1.5,
            )

        else:

            weights = (
                7.0,
                6.0,
                6.0,
                3.0,
                1.0,
                3.5,
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
            _ngrams(tokens, 4)
        )

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

                asin = str(
                    row[0]
                )

                phrase_weight = (
                    len(
                        phrase.split()
                    )
                    / 2.0
                )

                scores[asin] += (
                    phrase_weight
                    / (rank + 5.0)
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
        limit: int = 100,
    ) -> list[str]:

        category_terms = [
            token
            for token in tokens
            if token in CATEGORY_TERMS
        ]

        if not category_terms:
            return []

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
    # RRF
    # =========================================================

    def _rrf(
        self,
        routes: list[list[str]],
        weights: list[float],
    ) -> dict[str, float]:

        scores: dict[str, float] = {}

        k = 45.0

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
                        0.0,
                    )
                    + weight
                    / (k + rank)
                )

        return scores

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
    # CONSTRAINT-AWARE RERANKING
    # =========================================================

    def _rerank(
        self,
        candidates: list[str],
        current_message: str,
        tokens: list[str],
        state: dict,
        intent: str,
        top_k: int,
    ) -> list[tuple[float, str]]:

        current_tokens = set(
            _terms(
                current_message
            )
        )

        query_set = set(
            tokens
        )

        current_text = (
            " ".join(
                _terms(
                    current_message
                )
            )
        )

        query_2grams = set(
            _ngrams(
                list(query_set),
                2,
            )
        )

        query_3grams = set(
            _ngrams(
                list(query_set),
                3,
            )
        )

        constraints = state[
            "attributes"
        ]

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

            details = metadata[
                "details"
            ]

            description = metadata[
                "description"
            ]

            combined = (
                title
                + " "
                + categories
                + " "
                + features
                + " "
                + details
                + " "
                + description
            )

            score = 0.0

            current_matches = sum(
                1
                for token in current_tokens
                if token in combined
            )

            if current_tokens:

                current_coverage = (
                    current_matches
                    / len(current_tokens)
                )

                score += (
                    0.45
                    * current_coverage
                )

            title_tokens = set(
                _terms(title)
            )

            title_matches = (
                current_tokens
                & title_tokens
            )

            if current_tokens:

                title_coverage = (
                    len(title_matches)
                    / len(current_tokens)
                )

                score += (
                    0.75
                    * title_coverage
                )

            category_matches = (
                current_tokens
                & set(
                    _terms(categories)
                )
            )

            score += (
                0.25
                * len(category_matches)
            )

            if (
                current_text
                and current_text in title
            ):
                score += 1.25

            elif (
                current_text
                and current_text in combined
            ):
                score += 0.55

            for phrase in query_3grams:

                if phrase in title:
                    score += 0.22

                elif phrase in combined:
                    score += 0.08

            for phrase in query_2grams:

                if phrase in title:
                    score += 0.10

                elif phrase in combined:
                    score += 0.04

            for attribute_type in (
                "colors",
                "materials",
                "sizes",
            ):

                values = constraints[
                    attribute_type
                ]

                for value in values:

                    if value in title:

                        if intent == "buying":
                            score += 0.65
                        else:
                            score += 0.25

                    elif value in combined:

                        if intent == "buying":
                            score += 0.30
                        else:
                            score += 0.12

                    else:

                        if intent == "buying":
                            score -= 0.45

            categories_required = constraints[
                "categories"
            ]

            for category in categories_required:

                if category in title:
                    score += 0.40

                elif category in combined:
                    score += 0.20

                elif intent == "buying":
                    score -= 0.20

            score += (
                0.025
                * self._rating_score(
                    asin
                )
            )

            score += (
                0.03
                / (1.0 + base_rank)
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

        return scored[
            :max(
                top_k * 8,
                50,
            )
        ]

    # =========================================================
    # CLARIFICATION CHECK (disabled — see ENABLE_CLARIFICATION)
    # =========================================================

    def _needs_clarification(
        self,
        scored_candidates: list[tuple[float, str]],
        state: dict,
        intent: str,
        turn: int,
    ) -> str | None:

        if not self.ENABLE_CLARIFICATION:
            return None

        if turn >= 8:
            return None

        if intent != "buying":
            return None

        if len(scored_candidates) < 20:
            return None

        top_scores = [
            score for score, _ in scored_candidates[:10]
        ]

        if not top_scores:
            return None

        spread = top_scores[0] - top_scores[-1]

        if spread > 0.35:
            return None

        constraints = state["attributes"]

        if not constraints["categories"]:
            return "category"

        if not constraints["sizes"]:
            return "size"

        if not constraints["colors"]:
            return "color"

        if not constraints["materials"]:
            return "material"

        return None

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

        if intent == "buying":

            return candidates[
                :top_k
            ]

        lambda_value = 0.68

        protected_count = min(
            3,
            top_k,
        )

        selected = list(
            candidates[:protected_count]
        )

        query_set = set(
            tokens
        )

        candidate_sets = {}

        for asin in candidates:

            metadata = self._metadata.get(
                asin,
                {},
            )

            text = (
                metadata.get(
                    "title",
                    "",
                )
                + " "
                + metadata.get(
                    "features",
                    "",
                )
                + " "
                + metadata.get(
                    "categories",
                    "",
                )
            )

            candidate_sets[asin] = set(
                _terms(text)
            )

        remaining = [
            asin
            for asin in candidates
            if asin not in selected
        ]

        while (
            remaining
            and len(selected) < top_k
        ):

            best_asin = None
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

                similarities = []

                for chosen in selected:

                    a = candidate_set
                    b = candidate_sets[
                        chosen
                    ]

                    union = len(
                        a | b
                    )

                    if union == 0:
                        similarity = 0.0
                    else:
                        similarity = (
                            len(
                                a & b
                            )
                            / union
                        )

                    similarities.append(
                        similarity
                    )

                diversity_penalty = (
                    max(similarities)
                    if similarities
                    else 0.0
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

                if (
                    mmr_score
                    > best_score
                ):

                    best_score = (
                        mmr_score
                    )

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

        if (
            session_id
            not in self._sessions
        ):

            raise RuntimeError(
                "reset must be called before respond"
            )

        state = self._sessions[
            session_id
        ]

        lower_message = (
            user_message.lower()
        )

        # =====================================================
        # INTENT OVERRIDE
        # =====================================================

        # Discourse marker alone is enough to trigger an
        # override, matching V7. What changed is what gets
        # cleared: attributes always clear on override,
        # category only clears if a new category term is
        # actually present (see _update_state).
        is_override = (
            "ignore my earlier" in lower_message
            or "ignore earlier" in lower_message
            or "change that" in lower_message
            or "actually" in lower_message
            or "instead" in lower_message
            or "rather" in lower_message
        )

        message_tokens = set(
            _terms(user_message)
        )

        new_category_tokens = (
            message_tokens
            & CATEGORY_TERMS
        )

        existing_categories = state[
            "attributes"
        ]["categories"]

        override_category = bool(
            new_category_tokens
            and new_category_tokens
            != existing_categories
        )

        if is_override:

            state[
                "override_active"
            ] = True

        # =====================================================
        # UPDATE STATE
        # =====================================================

        self._update_state(
            state,
            user_message,
            is_override,
            override_category,
        )

        state[
            "history"
        ].append(
            user_message
        )

        # =====================================================
        # INTENT
        # =====================================================

        intent = self._detect_intent(
            user_message,
            state["history"][:-1],
        )

        state[
            "intent"
        ] = intent

        # =====================================================
        # QUERY
        # =====================================================

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
                    limit=120,
                )
            )

            category_route = (
                self._category_candidates(
                    tokens,
                    limit=120,
                )
            )

            if intent == "buying":

                routes = [
                    bm25_route,
                    phrase_route,
                    category_route,
                ]

                weights = [
                    0.70,
                    0.22,
                    0.08,
                ]

            else:

                routes = [
                    bm25_route,
                    category_route,
                    phrase_route,
                ]

                weights = [
                    0.55,
                    0.30,
                    0.15,
                ]

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
                )[:180]
            ]

            reranked_scored = self._rerank(
                candidates,
                user_message,
                tokens,
                state,
                intent,
                top_k,
            )

            clarify_slot = self._needs_clarification(
                reranked_scored,
                state,
                intent,
                turn,
            )

            if clarify_slot:

                return {
                    "message": (
                        f"To narrow this down, what {clarify_slot} "
                        "are you looking for?"
                    ),

                    "ask_attribute": clarify_slot,

                    "recommendations": [],

                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    },
                }

            reranked = [
                asin
                for _, asin in reranked_scored
            ]

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
                for asin in recommendations[
                    :top_k
                ]
            ],

            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }