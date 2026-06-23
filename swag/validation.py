from __future__ import annotations

from typing import Any

from swag.settings import settings
from swag.schemas import SemanticCategory, SWAGProfile, WeightedSemanticTerm
from swag.utils import term_grounded_in_text

LIST_KEYS = (
    "primary_concepts",
    "domain_terms",
    "direct_synonyms",
    "adjacent_concepts",
    "business_goals",
    "user_intents",
    "negative_contexts",
)


def _clip_str(value: object, max_len: int) -> str:
    text = str(value or "").strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _sanitize_string_list(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    cleaned: list[str] = []
    for item in items[: settings.swag_max_terms_per_list]:
        term = _clip_str(item, settings.swag_max_term_chars)
        if term and term not in cleaned:
            cleaned.append(term)
    return cleaned


def _sanitize_weighted_terms(
    raw_terms: object,
    *,
    chunk_text: str | None,
) -> list[WeightedSemanticTerm]:
    if not isinstance(raw_terms, list):
        return []

    terms: list[WeightedSemanticTerm] = []
    for item in raw_terms[: settings.swag_max_weighted_terms]:
        if not isinstance(item, dict):
            continue
        category_str = str(item.get("category", "primary_concept"))
        try:
            category = SemanticCategory(category_str)
        except ValueError:
            category = SemanticCategory.PRIMARY_CONCEPT

        term = _clip_str(item.get("term", ""), settings.swag_max_term_chars)
        if not term:
            continue

        weight = max(0.0, min(1.0, float(item.get("weight", 0.5))))
        if chunk_text and category != SemanticCategory.NEGATIVE_CONTEXT:
            if not term_grounded_in_text(term, chunk_text):
                weight *= settings.swag_ungrounded_term_weight_multiplier

        terms.append(
            WeightedSemanticTerm(
                term=term,
                category=category,
                weight=weight,
                reason=_clip_str(item.get("reason", ""), settings.swag_max_reason_chars),
            )
        )
    return terms


def sanitize_profile_data(
    data: dict[str, Any],
    *,
    chunk_text: str | None = None,
) -> dict[str, Any]:
    """Cap and validate LLM profile JSON; discount ungrounded chunk terms."""
    sanitized: dict[str, Any] = {
        "detected_domain": _clip_str(data.get("detected_domain", "general"), settings.swag_max_domain_chars),
        "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        "chunk_kind": _clip_str(data.get("chunk_kind", "body"), 32) or "body",
    }

    for key in LIST_KEYS:
        sanitized[key] = _sanitize_string_list(data.get(key, []))

    if chunk_text:
        for key in LIST_KEYS:
            if key == "negative_contexts":
                continue
            grounded: list[str] = []
            for term in sanitized[key]:
                if term_grounded_in_text(term, chunk_text):
                    grounded.append(term)
            sanitized[key] = grounded

    sanitized["weighted_terms"] = _sanitize_weighted_terms(
        data.get("weighted_terms", []),
        chunk_text=chunk_text,
    )
    return sanitized
