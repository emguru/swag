"""Build SWAG profiles from text — no LLM. Used for tests and low-cost demo."""

from __future__ import annotations

import re
from collections import Counter

from swag.chunk_quality import classify_chunk_kind
from swag.schemas import SWAGProfile, SemanticCategory, WeightedSemanticTerm

_STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off
    over under again further once here there all each few more most other some
    such no nor not only own same so than too very can will just don should now
    is are was were be been being have has had do does did doing what which who
    how why this that these those it its of as
    """.split()
)

_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "artificial_intelligence / natural_language_processing": (
        "rag",
        "retrieval",
        "augmented",
        "generation",
        "llm",
        "embedding",
        "vector",
        "chunk",
        "prompt",
        "language model",
    ),
    "finance": ("invoice", "payment", "vendor", "ledger", "accounting", "payroll"),
    "technology": ("software", "api", "cloud", "kubernetes", "database", "server"),
}

# Retrieval hints only — weighted and grounded at score time, not treated as facts.
_QUERY_TERM_HINTS: dict[str, list[tuple[str, SemanticCategory, float]]] = {
    "rag": [
        ("retrieval augmented generation", SemanticCategory.DIRECT_SYNONYM, 0.55),
        ("retrieval", SemanticCategory.ADJACENT_CONCEPT, 0.45),
    ],
    "useful": [
        ("benefit", SemanticCategory.BUSINESS_GOAL, 0.5),
        ("value", SemanticCategory.BUSINESS_GOAL, 0.48),
        ("advantage", SemanticCategory.ADJACENT_CONCEPT, 0.45),
    ],
    "worth": [
        ("benefit", SemanticCategory.BUSINESS_GOAL, 0.5),
        ("value", SemanticCategory.BUSINESS_GOAL, 0.48),
    ],
}

_EVALUATIVE_PATTERNS = (
    re.compile(r"\b(is|are)\s+.+\s+(useful|worth|beneficial|effective|better)\b", re.I),
    re.compile(r"\b(worth\s+(it|using|the effort))\b", re.I),
    re.compile(r"\b(benefits?|advantages?|value)\s+of\b", re.I),
    re.compile(r"^(what are|what is)\s+.+\s+(benefits|advantages|value)\b", re.I),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]


def _significant_terms(text: str, *, limit: int = 10) -> list[str]:
    words = [w for w in _tokens(text) if w not in _STOPWORDS and len(w) >= 3]
    counts = Counter(words)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], -len(x[0])))
    return [term for term, _ in ranked[:limit]]


def _phrases(text: str, *, limit: int = 4) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    phrases: list[str] = []
    if len(cleaned) <= 80:
        phrases.append(cleaned)
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        sentence = sentence.strip()
        if len(sentence) >= 12 and sentence not in phrases:
            phrases.append(sentence[:80])
        if len(phrases) >= limit:
            break
    return phrases[:limit]


def _guess_domain(text: str) -> str:
    lower = text.lower()
    best_domain = "general"
    best_hits = 0
    for domain, hints in _DOMAIN_HINTS.items():
        hits = sum(1 for hint in hints if hint in lower)
        if hits > best_hits:
            best_hits = hits
            best_domain = domain
    return best_domain


def is_evaluative_query(query: str) -> bool:
    cleaned = query.strip()
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return True
    if any(cleaned.lower().startswith(w) for w in ("what", "how", "why", "when", "where", "who")):
        return True
    return any(pattern.search(cleaned) for pattern in _EVALUATIVE_PATTERNS)


def _weighted_from_terms(
    terms: list[str],
    category: SemanticCategory,
    *,
    base_weight: float,
) -> list[WeightedSemanticTerm]:
    weighted: list[WeightedSemanticTerm] = []
    for idx, term in enumerate(terms):
        weight = max(0.4, base_weight - idx * 0.05)
        weighted.append(
            WeightedSemanticTerm(
                term=term,
                category=category,
                weight=weight,
                reason="heuristic extraction",
            )
        )
    return weighted


def _query_expansion_hints(query: str) -> list[WeightedSemanticTerm]:
    """Low-weight retrieval hints from query tokens — not authoritative synonyms."""
    hints: list[WeightedSemanticTerm] = []
    seen: set[str] = set()
    for token in _tokens(query):
        if token in _STOPWORDS or len(token) < 3:
            continue
        for hint_term, category, weight in _QUERY_TERM_HINTS.get(token, ()):
            key = hint_term.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(
                WeightedSemanticTerm(
                    term=hint_term,
                    category=category,
                    weight=weight,
                    reason="query expansion hint",
                )
            )
    return hints


def _off_topic_negative_contexts(domain: str, *, limit: int = 8) -> list[str]:
    """Terms from non-matching domains — used as query negatives to reduce cross-topic noise."""
    negatives: list[str] = []
    for other_domain, hints in _DOMAIN_HINTS.items():
        if other_domain == domain:
            continue
        negatives.extend(hints[:2])
    return negatives[:limit]


def heuristic_query_profile(query: str) -> SWAGProfile:
    query = query.strip()
    terms = _significant_terms(query, limit=8)
    primary = [query] if query else []
    domain = _guess_domain(query)
    negatives = _off_topic_negative_contexts(domain)
    lower = query.lower()
    if any(token in lower for token in ("basic", "basics", "fundamental", "introduction", "overview")):
        negatives.extend(
            [
                "incident reporting",
                "adverse event",
                "openfda",
                "enforcement action",
                "recall notice",
            ]
        )
    if any(token in lower for token in ("allergen", "labeling", "menu", "self service")):
        negatives.extend(["incident reporting", "openfda", "adverse event"])
    evaluative = is_evaluative_query(query)

    weighted: list[WeightedSemanticTerm] = []
    weighted.extend(_weighted_from_terms(primary, SemanticCategory.PRIMARY_CONCEPT, base_weight=0.95))
    weighted.extend(_weighted_from_terms(terms, SemanticCategory.DOMAIN_TERM, base_weight=0.8))
    weighted.extend(_query_expansion_hints(query))

    direct_synonyms: list[str] = []
    adjacent: list[str] = []
    business_goals: list[str] = []
    for hint in weighted:
        if hint.reason != "query expansion hint":
            continue
        if hint.category == SemanticCategory.DIRECT_SYNONYM:
            direct_synonyms.append(hint.term)
        elif hint.category == SemanticCategory.ADJACENT_CONCEPT:
            adjacent.append(hint.term)
        elif hint.category == SemanticCategory.BUSINESS_GOAL:
            business_goals.append(hint.term)

    user_intents: list[str] = []
    if evaluative and query:
        user_intents.append(query)
        if "useful" in query.lower() or "worth" in query.lower():
            business_goals.extend(["assess usefulness", "evaluate value"])

    return SWAGProfile(
        primary_concepts=primary[:4],
        domain_terms=terms[:8],
        direct_synonyms=direct_synonyms[:6],
        adjacent_concepts=adjacent[:6],
        business_goals=business_goals[:6],
        user_intents=user_intents[:2],
        negative_contexts=negatives,
        detected_domain=domain,
        confidence=0.8 if evaluative else 0.75,
        weighted_terms=weighted[:24],
    )


def heuristic_chunk_profile(
    chunk_text: str,
    *,
    source_filename: str = "unknown",
) -> SWAGProfile:
    text = chunk_text.strip()
    chunk_kind = classify_chunk_kind(text)
    primary = _phrases(text, limit=3) or ([text[:80]] if text else [])
    domain_terms = _significant_terms(text, limit=10)
    filename_terms = _significant_terms(source_filename.replace("_", " ").replace("-", " "), limit=3)
    for term in filename_terms:
        if term not in domain_terms:
            domain_terms.append(term)

    weighted: list[WeightedSemanticTerm] = []
    weighted.extend(_weighted_from_terms(primary, SemanticCategory.PRIMARY_CONCEPT, base_weight=0.9))
    weighted.extend(_weighted_from_terms(domain_terms, SemanticCategory.DOMAIN_TERM, base_weight=0.75))

    confidence = 0.7 if text else 0.0
    if chunk_kind in {"boilerplate", "empty"}:
        confidence = 0.15
    elif chunk_kind in {"title", "nav"}:
        confidence = 0.45

    return SWAGProfile(
        primary_concepts=primary[:6],
        domain_terms=domain_terms[:10],
        detected_domain=_guess_domain(text + " " + source_filename),
        chunk_kind=chunk_kind,
        confidence=confidence,
        weighted_terms=weighted[:24],
    )
