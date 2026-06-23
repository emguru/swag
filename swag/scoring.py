from __future__ import annotations

from swag.settings import settings
from swag.schemas import SemanticCategory, SWAGProfile
from swag.utils import (
    fuzzy_term_match,
    negative_penalty_match,
    term_grounded_in_text,
    term_mentioned_contrastively,
)

CATEGORY_WEIGHTS: dict[SemanticCategory, float] = {
    SemanticCategory.PRIMARY_CONCEPT: 1.0,
    SemanticCategory.DOMAIN_TERM: 0.85,
    SemanticCategory.DIRECT_SYNONYM: 0.70,
    SemanticCategory.ADJACENT_CONCEPT: 0.60,
    SemanticCategory.BUSINESS_GOAL: 0.75,
    SemanticCategory.USER_INTENT: 0.90,
    SemanticCategory.NEGATIVE_CONTEXT: 0.0,
}

NEGATIVE_MATCH_PENALTY = 0.4


def _extract_negative_terms(profile: SWAGProfile) -> list[str]:
    negatives = list(profile.negative_contexts)
    if profile.weighted_terms:
        for wt in profile.weighted_terms:
            if wt.category == SemanticCategory.NEGATIVE_CONTEXT and wt.term not in negatives:
                negatives.append(wt.term)
    return negatives


def _extract_positive_query_terms(
    query_profile: SWAGProfile,
) -> list[tuple[str, SemanticCategory, float]]:
    if query_profile.weighted_terms:
        return [
            (wt.term, wt.category, wt.weight)
            for wt in query_profile.weighted_terms
            if wt.category != SemanticCategory.NEGATIVE_CONTEXT
        ]

    terms: list[tuple[str, SemanticCategory, float]] = []
    for concept in query_profile.primary_concepts:
        terms.append((concept, SemanticCategory.PRIMARY_CONCEPT, 0.9))
    for term in query_profile.domain_terms:
        terms.append((term, SemanticCategory.DOMAIN_TERM, 0.8))
    for syn in query_profile.direct_synonyms:
        terms.append((syn, SemanticCategory.DIRECT_SYNONYM, 0.7))
    for adj in query_profile.adjacent_concepts:
        terms.append((adj, SemanticCategory.ADJACENT_CONCEPT, 0.6))
    for goal in query_profile.business_goals:
        terms.append((goal, SemanticCategory.BUSINESS_GOAL, 0.75))
    for intent in query_profile.user_intents:
        terms.append((intent, SemanticCategory.USER_INTENT, 0.85))
    return terms


def _extract_chunk_terms(chunk_profile: SWAGProfile) -> list[tuple[str, SemanticCategory, float]]:
    if chunk_profile.weighted_terms:
        return [
            (wt.term, wt.category, wt.weight)
            for wt in chunk_profile.weighted_terms
            if wt.category != SemanticCategory.NEGATIVE_CONTEXT
        ]

    terms: list[tuple[str, SemanticCategory, float]] = []
    for concept in chunk_profile.primary_concepts:
        terms.append((concept, SemanticCategory.PRIMARY_CONCEPT, 0.9))
    for term in chunk_profile.domain_terms:
        terms.append((term, SemanticCategory.DOMAIN_TERM, 0.8))
    for syn in chunk_profile.direct_synonyms:
        terms.append((syn, SemanticCategory.DIRECT_SYNONYM, 0.7))
    for adj in chunk_profile.adjacent_concepts:
        terms.append((adj, SemanticCategory.ADJACENT_CONCEPT, 0.6))
    for goal in chunk_profile.business_goals:
        terms.append((goal, SemanticCategory.BUSINESS_GOAL, 0.75))
    for intent in chunk_profile.user_intents:
        terms.append((intent, SemanticCategory.USER_INTENT, 0.85))
    return terms


def _chunk_term_grounding_multiplier(term: str, chunk_text: str) -> float:
    if not chunk_text:
        return 1.0
    if term_grounded_in_text(term, chunk_text):
        return 1.0
    return settings.swag_ungrounded_term_weight_multiplier


def compute_metadata_boost(
    *,
    chunk_profile: SWAGProfile | None,
    query_profile: SWAGProfile,
    source: str = "",
    original_query: str = "",
) -> float:
    boost = 0.0
    if chunk_profile:
        if (
            chunk_profile.detected_domain
            and query_profile.detected_domain
            and chunk_profile.detected_domain.lower() == query_profile.detected_domain.lower()
        ):
            boost += 0.5
        boost += chunk_profile.confidence * 0.3

    query_words = set(original_query.lower().split()) if original_query else set()
    if query_words and source:
        filename_lower = source.lower()
        min_len = 4
        overlap = sum(
            1 for word in query_words if len(word) >= min_len and word in filename_lower
        )
        boost += min(overlap * 0.1, 0.2)

    return min(boost, 1.0)


def _category_weight(category: SemanticCategory, *, evaluative_query: bool) -> float:
    base = CATEGORY_WEIGHTS.get(category, 0.5)
    if evaluative_query and category in {
        SemanticCategory.USER_INTENT,
        SemanticCategory.BUSINESS_GOAL,
    }:
        return min(1.0, base * settings.swag_intent_category_boost)
    return base


def score_swag_alignment(
    query_profile: SWAGProfile,
    chunk_profile: SWAGProfile,
    *,
    chunk_text: str = "",
    evaluative_query: bool = False,
) -> tuple[float, list[str]]:
    if not chunk_profile or chunk_profile.confidence == 0.0:
        return 0.0, ["No SWAG profile available for chunk"]

    query_terms = _extract_positive_query_terms(query_profile)
    chunk_terms = _extract_chunk_terms(chunk_profile)
    query_negatives = _extract_negative_terms(query_profile)
    chunk_negatives = _extract_negative_terms(chunk_profile)

    if not query_terms or not chunk_terms:
        return 0.0, ["Insufficient semantic terms for comparison"]

    positive_score = 0.0
    max_possible = 0.0
    explanations: list[str] = []
    negative_hits = 0

    chunk_term_map: dict[str, list[tuple[SemanticCategory, float]]] = {}
    for term, category, weight in chunk_terms:
        chunk_term_map.setdefault(term.lower(), []).append((category, weight))

    for q_term, q_category, q_weight in query_terms:
        cat_weight = _category_weight(q_category, evaluative_query=evaluative_query)
        term_max = q_weight * cat_weight
        max_possible += term_max

        best_match = 0.0
        best_reason = ""
        for c_term, entries in chunk_term_map.items():
            if not fuzzy_term_match(q_term, c_term):
                continue
            grounding = _chunk_term_grounding_multiplier(c_term, chunk_text)
            for c_category, c_weight in entries:
                c_cat_weight = _category_weight(c_category, evaluative_query=evaluative_query)
                match_score = (
                    q_weight * c_weight * min(cat_weight, c_cat_weight) * grounding
                )
                if match_score > best_match:
                    best_match = match_score
                    reason = (
                        f"Query '{q_term}' ({q_category.value}) aligned with "
                        f"chunk '{c_term}' ({c_category.value}, w={c_weight:.2f})"
                    )
                    if grounding < 1.0:
                        reason += " [ungrounded term discounted]"
                    best_reason = reason

        if best_match > 0:
            positive_score += best_match
            explanations.append(best_reason)

        for neg_term in chunk_negatives:
            if fuzzy_term_match(q_term, neg_term):
                negative_hits += 1
                explanations.append(
                    f"Penalty: query term '{q_term}' matches chunk negative context '{neg_term}'"
                )

    for q_neg in query_negatives:
        for c_term, entries in chunk_term_map.items():
            if not negative_penalty_match(q_neg, c_term):
                continue
            if term_mentioned_contrastively(c_term, chunk_text):
                explanations.append(
                    f"Skipped penalty: chunk term '{c_term}' appears contrastively in text "
                    f"for query negative '{q_neg}'"
                )
                continue
            negative_hits += 1
            categories = ", ".join(cat.value for cat, _ in entries)
            explanations.append(
                f"Penalty: query negative context '{q_neg}' matches chunk term "
                f"'{c_term}' ({categories})"
            )

    if (
        query_profile.detected_domain != "general"
        and chunk_profile.detected_domain != "general"
        and query_profile.detected_domain.lower() != chunk_profile.detected_domain.lower()
    ):
        positive_score *= 0.7
        explanations.append(
            f"Domain mismatch: query={query_profile.detected_domain}, "
            f"chunk={chunk_profile.detected_domain}"
        )

    normalized = positive_score / max_possible if max_possible > 0 else 0.0
    if negative_hits > 0:
        normalized = max(0.0, normalized - NEGATIVE_MATCH_PENALTY * negative_hits)

    return min(1.0, max(0.0, normalized)), explanations


def compute_final_score(
    vector_similarity: float,
    swag_score: float,
    metadata_boost: float,
    *,
    vector_weight: float,
    swag_weight: float,
    metadata_weight: float,
) -> float:
    return (
        vector_similarity * vector_weight
        + swag_score * swag_weight
        + metadata_boost * metadata_weight
    )


def build_selection_explanation(
    vector_similarity: float,
    swag_score: float,
    metadata_boost: float,
    final_score: float,
    swag_explanations: list[str],
) -> str:
    parts = [
        f"Final score {final_score:.3f} = "
        f"vector({vector_similarity:.3f}) + swag({swag_score:.3f}) + "
        f"metadata({metadata_boost:.3f})."
    ]
    if swag_explanations:
        parts.append("SWAG matches: " + "; ".join(swag_explanations[:5]))
        if len(swag_explanations) > 5:
            parts.append(f"(+{len(swag_explanations) - 5} more matches)")
    return " ".join(parts)
