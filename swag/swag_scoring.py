"""
SWAG scoring — compare query profile against chunk SWAG profiles.

Produces a relevance score for reranking without modifying original chunk content.
"""

from __future__ import annotations

import logging
from typing import Optional

from swag.schemas import (
    DocumentChunk,
    QueryProfile,
    SemanticCategory,
    SWAGProfile,
    WeightedSemanticTerm,
)
from swag.utils import fuzzy_term_match

logger = logging.getLogger(__name__)

# Category weights for positive matching (higher = more important for retrieval)
CATEGORY_WEIGHTS: dict[SemanticCategory, float] = {
    SemanticCategory.PRIMARY_CONCEPT: 1.0,
    SemanticCategory.DOMAIN_TERM: 0.85,
    SemanticCategory.DIRECT_SYNONYM: 0.70,
    SemanticCategory.ADJACENT_CONCEPT: 0.60,
    SemanticCategory.BUSINESS_GOAL: 0.75,
    SemanticCategory.USER_INTENT: 0.90,
    SemanticCategory.NEGATIVE_CONTEXT: 0.0,  # handled separately as penalty
}

# Penalty multiplier when query matches chunk negative contexts
NEGATIVE_MATCH_PENALTY = 0.4


def _extract_query_terms(query_profile: SWAGProfile) -> list[tuple[str, SemanticCategory, float]]:
    """Flatten query profile into (term, category, weight) tuples."""
    terms: list[tuple[str, SemanticCategory, float]] = []

    if query_profile.weighted_terms:
        for wt in query_profile.weighted_terms:
            terms.append((wt.term, wt.category, wt.weight))
    else:
        # Fallback to list fields with default weights
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
    """Flatten chunk profile into (term, category, weight) tuples."""
    if chunk_profile.weighted_terms:
        return [(wt.term, wt.category, wt.weight) for wt in chunk_profile.weighted_terms]

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
    for neg in chunk_profile.negative_contexts:
        terms.append((neg, SemanticCategory.NEGATIVE_CONTEXT, 1.0))
    return terms


def compute_metadata_boost(
    chunk: DocumentChunk,
    query_profile: SWAGProfile,
    original_query: str = "",
) -> float:
    """
    Small boost from structural metadata: domain alignment, page presence, confidence.
    """
    boost = 0.0
    chunk_profile = chunk.swag_profile

    if chunk_profile:
        # Domain alignment
        if (
            chunk_profile.detected_domain
            and query_profile.detected_domain
            and chunk_profile.detected_domain.lower() == query_profile.detected_domain.lower()
        ):
            boost += 0.5

        # Profile confidence contributes slightly
        boost += chunk_profile.confidence * 0.3

    # Source filename relevance (simple keyword check)
    query_words = set(original_query.lower().split()) if original_query else set()
    if query_words:
        filename_lower = chunk.source_filename.lower()
        overlap = sum(1 for w in query_words if w in filename_lower and len(w) > 3)
        boost += min(overlap * 0.1, 0.2)

    return min(boost, 1.0)


def score_swag_alignment(
    query_profile: SWAGProfile,
    chunk_profile: SWAGProfile,
) -> tuple[float, list[str]]:
    """
    Compare query SWAG profile against chunk SWAG profile.

    Returns (score 0-1, list of match explanations).
    """
    if not chunk_profile or chunk_profile.confidence == 0.0:
        return 0.0, ["No SWAG profile available for chunk"]

    query_terms = _extract_query_terms(query_profile)
    chunk_terms = _extract_chunk_terms(chunk_profile)

    if not query_terms or not chunk_terms:
        return 0.0, ["Insufficient semantic terms for comparison"]

    positive_score = 0.0
    max_possible = 0.0
    explanations: list[str] = []
    negative_hits = 0

    chunk_term_map: dict[str, list[tuple[SemanticCategory, float]]] = {}
    chunk_negatives: list[str] = []

    for term, category, weight in chunk_terms:
        if category == SemanticCategory.NEGATIVE_CONTEXT:
            chunk_negatives.append(term)
        else:
            chunk_term_map.setdefault(term.lower(), []).append((category, weight))

    for q_term, q_category, q_weight in query_terms:
        cat_weight = CATEGORY_WEIGHTS.get(q_category, 0.5)
        term_max = q_weight * cat_weight
        max_possible += term_max

        best_match = 0.0
        best_reason = ""

        for c_term, entries in chunk_term_map.items():
            if fuzzy_term_match(q_term, c_term):
                for c_category, c_weight in entries:
                    c_cat_weight = CATEGORY_WEIGHTS.get(c_category, 0.5)
                    match_score = q_weight * c_weight * min(cat_weight, c_cat_weight)
                    if match_score > best_match:
                        best_match = match_score
                        best_reason = (
                            f"Query '{q_term}' ({q_category.value}) aligned with "
                            f"chunk '{c_term}' ({c_category.value}, w={c_weight:.2f})"
                        )

        if best_match > 0:
            positive_score += best_match
            explanations.append(best_reason)

        # Check negative context penalty
        for neg_term in chunk_negatives:
            if fuzzy_term_match(q_term, neg_term):
                negative_hits += 1
                explanations.append(
                    f"Penalty: query term '{q_term}' matches negative context '{neg_term}'"
                )

    # Domain mismatch penalty
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

    # Normalize to 0-1
    if max_possible > 0:
        normalized = positive_score / max_possible
    else:
        normalized = 0.0

    # Apply negative context penalty
    if negative_hits > 0:
        penalty = NEGATIVE_MATCH_PENALTY * negative_hits
        normalized = max(0.0, normalized - penalty)

    normalized = min(1.0, max(0.0, normalized))
    return normalized, explanations


def compute_final_score(
    vector_similarity: float,
    swag_score: float,
    metadata_boost: float,
    vector_weight: float = 0.65,
    swag_weight: float = 0.30,
    metadata_weight: float = 0.05,
) -> float:
    """
    Combine scores using configurable formula:
    final = vector * VECTOR_WEIGHT + swag * SWAG_WEIGHT + metadata * METADATA_WEIGHT
    """
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
    """Human-readable explanation of why a chunk was selected."""
    parts = [
        f"Final score {final_score:.3f} = "
        f"vector({vector_similarity:.3f}) + swag({swag_score:.3f}) + metadata({metadata_boost:.3f})."
    ]
    if swag_explanations:
        parts.append("SWAG matches: " + "; ".join(swag_explanations[:5]))
        if len(swag_explanations) > 5:
            parts.append(f"(+{len(swag_explanations) - 5} more matches)")
    return " ".join(parts)
