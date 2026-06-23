from __future__ import annotations

import json
import logging
from typing import Any

from swag.settings import settings
from swag.schemas import SWAGProfile, SemanticCategory, WeightedSemanticTerm
from swag.heuristic import heuristic_chunk_profile, heuristic_query_profile
from swag.validation import sanitize_profile_data

logger = logging.getLogger(__name__)

SWAG_SYSTEM_PROMPT = """You are a semantic analysis engine for a retrieval system called SWAG
(Semantic Weighted Augmented Generation).

Your job is to analyze document chunks and produce STRUCTURED METADATA for reranking.
This metadata is NOT source content — it must NEVER replace or modify the original text.

CRITICAL RULES:
1. NEVER treat generated synonyms as facts about the document.
2. NEVER expand isolated ambiguous words without full chunk context.
3. Expand CONCEPTS and PHRASES, not single keywords in isolation.
4. Detect the domain FIRST (e.g., finance, healthcare, technology, legal).
5. Include negative_contexts — terms that could cause FALSE matches in other domains.
6. Assign weights 0.0-1.0 reflecting contextual relevance to THIS chunk only.
7. direct_synonyms are retrieval hints only — not authoritative definitions.

Return valid JSON matching the requested schema."""

SWAG_CHUNK_USER_TEMPLATE = """Analyze this document chunk and produce a SWAG semantic profile.

CHUNK TEXT:
{chunk_text}

SOURCE FILE: {source_filename}
PAGE: {page_number}

Return JSON with this exact structure:
{{
  "detected_domain": "string",
  "primary_concepts": [],
  "domain_terms": [],
  "direct_synonyms": [],
  "adjacent_concepts": [],
  "business_goals": [],
  "user_intents": [],
  "negative_contexts": [],
  "confidence": 0.0,
  "weighted_terms": [
    {{
      "term": "example term",
      "category": "primary_concept|domain_term|direct_synonym|adjacent_concept|business_goal|user_intent|negative_context",
      "weight": 0.0,
      "reason": "brief explanation tied to chunk context"
    }}
  ]
}}"""

SWAG_QUERY_USER_TEMPLATE = """Analyze this user query and produce a SWAG semantic profile for retrieval matching.

QUERY:
{query}

Return JSON with this exact structure:
{{
  "detected_domain": "string",
  "primary_concepts": [],
  "domain_terms": [],
  "direct_synonyms": [],
  "adjacent_concepts": [],
  "business_goals": [],
  "user_intents": [],
  "negative_contexts": [],
  "confidence": 0.0,
  "weighted_terms": [
    {{
      "term": "example term",
      "category": "primary_concept|domain_term|direct_synonym|adjacent_concept|business_goal|user_intent|negative_context",
      "weight": 0.0,
      "reason": "brief explanation tied to query intent"
    }}
  ]
}}"""

CATEGORY_MAP = {
    "primary_concept": SemanticCategory.PRIMARY_CONCEPT,
    "domain_term": SemanticCategory.DOMAIN_TERM,
    "direct_synonym": SemanticCategory.DIRECT_SYNONYM,
    "adjacent_concept": SemanticCategory.ADJACENT_CONCEPT,
    "business_goal": SemanticCategory.BUSINESS_GOAL,
    "user_intent": SemanticCategory.USER_INTENT,
    "negative_context": SemanticCategory.NEGATIVE_CONTEXT,
}


def _complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    from swag.llm_client import create_llm_client

    return create_llm_client().complete_json(system_prompt, user_prompt)


def _parse_weighted_terms(raw_terms: list[dict[str, Any]]) -> list[WeightedSemanticTerm]:
    terms: list[WeightedSemanticTerm] = []
    for item in raw_terms:
        cat_str = item.get("category", "primary_concept")
        category = CATEGORY_MAP.get(cat_str, SemanticCategory.PRIMARY_CONCEPT)
        weight = max(0.0, min(1.0, float(item.get("weight", 0.5))))
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        terms.append(
            WeightedSemanticTerm(
                term=term,
                category=category,
                weight=weight,
                reason=str(item.get("reason", "")),
            )
        )
    return terms


def _normalize_profile_json(data: dict[str, Any]) -> dict[str, Any]:
    """Map alternate LLM key names onto the canonical SWAG profile schema."""
    normalized = dict(data)
    if not normalized.get("primary_concepts"):
        for alt in ("core_concepts", "concepts", "key_concepts"):
            if normalized.get(alt):
                normalized["primary_concepts"] = list(normalized[alt])
                break
    if not normalized.get("user_intents") and normalized.get("intent"):
        intent = normalized.get("intent")
        normalized["user_intents"] = [intent] if isinstance(intent, str) else list(intent or [])
    if not normalized.get("domain_terms") and normalized.get("subdomain"):
        sub = normalized.get("subdomain")
        normalized["domain_terms"] = [sub] if isinstance(sub, str) else list(sub or [])
    return normalized


def _build_profile_from_json(
    data: dict[str, Any],
    *,
    chunk_text: str | None = None,
) -> SWAGProfile:
    data = sanitize_profile_data(_normalize_profile_json(data), chunk_text=chunk_text)
    weighted_raw = data.get("weighted_terms", [])
    if weighted_raw and isinstance(weighted_raw[0], WeightedSemanticTerm):
        weighted = list(weighted_raw)
    else:
        weighted = _parse_weighted_terms(weighted_raw)
    lists = {
        "primary_concepts": list(data.get("primary_concepts", [])),
        "domain_terms": list(data.get("domain_terms", [])),
        "direct_synonyms": list(data.get("direct_synonyms", [])),
        "adjacent_concepts": list(data.get("adjacent_concepts", [])),
        "business_goals": list(data.get("business_goals", [])),
        "user_intents": list(data.get("user_intents", [])),
        "negative_contexts": list(data.get("negative_contexts", [])),
    }

    cat_to_key = {
        SemanticCategory.PRIMARY_CONCEPT: "primary_concepts",
        SemanticCategory.DOMAIN_TERM: "domain_terms",
        SemanticCategory.DIRECT_SYNONYM: "direct_synonyms",
        SemanticCategory.ADJACENT_CONCEPT: "adjacent_concepts",
        SemanticCategory.BUSINESS_GOAL: "business_goals",
        SemanticCategory.USER_INTENT: "user_intents",
        SemanticCategory.NEGATIVE_CONTEXT: "negative_contexts",
    }
    for wt in weighted:
        list_key = cat_to_key.get(wt.category, "primary_concepts")
        if wt.term not in lists[list_key]:
            lists[list_key].append(wt.term)

    return SWAGProfile(
        primary_concepts=lists["primary_concepts"],
        domain_terms=lists["domain_terms"],
        direct_synonyms=lists["direct_synonyms"],
        adjacent_concepts=lists["adjacent_concepts"],
        business_goals=lists["business_goals"],
        user_intents=lists["user_intents"],
        negative_contexts=lists["negative_contexts"],
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
        detected_domain=str(data.get("detected_domain", "general")),
        weighted_terms=weighted,
    )


def profile_from_payload(raw: object) -> SWAGProfile | None:
    if not raw:
        return None
    if isinstance(raw, SWAGProfile):
        return raw
    if isinstance(raw, dict):
        try:
            return SWAGProfile.model_validate(raw)
        except Exception:
            return _build_profile_from_json(raw)
    return None


class SWAGProfileGenerator:
    def _use_heuristic(self) -> bool:
        return settings.swag_profile_mode.lower() == "heuristic"

    def generate_chunk_profile(
        self,
        chunk_text: str,
        source_filename: str = "unknown",
        page_number: int | None = None,
    ) -> SWAGProfile:
        if self._use_heuristic():
            return heuristic_chunk_profile(
                chunk_text,
                source_filename=source_filename,
            )
        user_prompt = SWAG_CHUNK_USER_TEMPLATE.format(
            chunk_text=chunk_text,
            source_filename=source_filename,
            page_number=page_number or "N/A",
        )
        try:
            data = _complete_json(SWAG_SYSTEM_PROMPT, user_prompt)
            return _build_profile_from_json(data, chunk_text=chunk_text)
        except Exception as exc:
            logger.warning("SWAG chunk profile generation failed: %s", exc)
            return SWAGProfile(confidence=0.0, detected_domain="unknown")

    def generate_query_profile(self, query: str) -> SWAGProfile:
        if self._use_heuristic():
            return heuristic_query_profile(query)
        user_prompt = SWAG_QUERY_USER_TEMPLATE.format(query=query)
        try:
            data = _complete_json(SWAG_SYSTEM_PROMPT, user_prompt)
            return _build_profile_from_json(data, chunk_text=None)
        except Exception as exc:
            logger.warning("SWAG query profile generation failed: %s", exc)
            return SWAGProfile(
                primary_concepts=[query],
                confidence=0.3,
                detected_domain="general",
            )
