"""
SWAG profile generation — semantic metadata for reranking only.

Guardrails:
- Do NOT append metadata to original chunk text
- Do NOT treat synonyms as source facts
- Expand concepts, not isolated ambiguous keywords
- Detect domain before generating metadata
- Include negative contexts to prevent false matches
"""

from __future__ import annotations

import logging
from typing import Optional

from swag.llm_client import BaseLLMClient, create_llm_client
from swag.schemas import (
    SemanticCategory,
    SWAGProfile,
    WeightedSemanticTerm,
)
from swag.config import SWAGConfig

logger = logging.getLogger(__name__)

SWAG_SYSTEM_PROMPT = """You are a semantic analysis engine for a retrieval system called SWAG
(Semantic Weighted Augmented Generation).

Your job is to analyze document chunks and produce STRUCTURED METADATA for reranking.
This metadata is NOT source content — it must NEVER replace or modify the original text.

CRITICAL RULES:
1. NEVER treat generated synonyms as facts about the document.
2. NEVER expand isolated ambiguous words without full chunk context.
   Example: "Apple" alone is ambiguous — use surrounding context to determine
   technology company vs fruit.
3. Expand CONCEPTS and PHRASES, not single keywords in isolation.
4. Detect the domain FIRST (e.g., finance, healthcare, technology, legal).
5. Include negative_contexts — terms that could cause FALSE matches in other domains.
   Example: For "Apple silicon" in tech context, negative_contexts include
   "fruit", "orchard", "agriculture".
6. Assign weights 0.0-1.0 reflecting contextual relevance to THIS chunk only.
7. direct_synonyms are retrieval hints only — not authoritative definitions.

Return valid JSON matching the requested schema."""

SWAG_CHUNK_USER_TEMPLATE = """Analyze this document chunk and produce a SWAG semantic profile.

DETECTED CONTEXT: Analyze the full chunk before assigning any terms.

CHUNK TEXT:
{chunk_text}

SOURCE FILE: {source_filename}
PAGE: {page_number}

Return JSON with this exact structure:
{{
  "detected_domain": "string — primary domain (e.g., technology, finance, healthcare)",
  "primary_concepts": ["core concepts explicitly discussed in the chunk"],
  "domain_terms": ["specialized vocabulary for the detected domain"],
  "direct_synonyms": ["synonymous phrases for retrieval ONLY — not facts"],
  "adjacent_concepts": ["closely related concepts not explicitly stated"],
  "business_goals": ["business objectives this content supports"],
  "user_intents": ["questions a user might ask that this chunk answers"],
  "negative_contexts": ["domains/terms that would cause FALSE positive matches"],
  "confidence": 0.0-1.0,
  "weighted_terms": [
    {{
      "term": "example term",
      "category": "primary_concept|domain_term|direct_synonym|adjacent_concept|business_goal|user_intent|negative_context",
      "weight": 0.0-1.0,
      "reason": "brief explanation tied to chunk context"
    }}
  ]
}}"""

SWAG_QUERY_USER_TEMPLATE = """Analyze this user query and produce a SWAG semantic profile for retrieval matching.

QUERY:
{query}

RULES:
- Expand the query into concepts, NOT isolated ambiguous keywords.
- Detect the likely domain/intent of the query.
- Include negative_contexts for terms that could cause false matches.

Return JSON with the same structure as chunk profiles:
{{
  "detected_domain": "string",
  "primary_concepts": [],
  "domain_terms": [],
  "direct_synonyms": [],
  "adjacent_concepts": [],
  "business_goals": [],
  "user_intents": [],
  "negative_contexts": [],
  "confidence": 0.0-1.0,
  "weighted_terms": [
    {{"term": "...", "category": "...", "weight": 0.0-1.0, "reason": "..."}}
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


def _parse_weighted_terms(raw_terms: list[dict]) -> list[WeightedSemanticTerm]:
    """Parse and validate weighted terms from LLM JSON."""
    terms: list[WeightedSemanticTerm] = []
    for item in raw_terms:
        cat_str = item.get("category", "primary_concept")
        category = CATEGORY_MAP.get(cat_str, SemanticCategory.PRIMARY_CONCEPT)
        weight = max(0.0, min(1.0, float(item.get("weight", 0.5))))
        term = item.get("term", "").strip()
        if not term:
            continue
        terms.append(
            WeightedSemanticTerm(
                term=term,
                category=category,
                weight=weight,
                reason=item.get("reason", ""),
            )
        )
    return terms


def _build_profile_from_json(data: dict) -> SWAGProfile:
    """Convert LLM JSON response to SWAGProfile."""
    weighted = _parse_weighted_terms(data.get("weighted_terms", []))

    # Sync list fields from weighted terms if lists are empty
    lists = {
        "primary_concepts": data.get("primary_concepts", []),
        "domain_terms": data.get("domain_terms", []),
        "direct_synonyms": data.get("direct_synonyms", []),
        "adjacent_concepts": data.get("adjacent_concepts", []),
        "business_goals": data.get("business_goals", []),
        "user_intents": data.get("user_intents", []),
        "negative_contexts": data.get("negative_contexts", []),
    }

    # Enrich lists from weighted terms
    for wt in weighted:
        key = wt.category.value + "s" if not wt.category.value.endswith("s") else wt.category.value
        # Map category enum to list key
        cat_to_key = {
            SemanticCategory.PRIMARY_CONCEPT: "primary_concepts",
            SemanticCategory.DOMAIN_TERM: "domain_terms",
            SemanticCategory.DIRECT_SYNONYM: "direct_synonyms",
            SemanticCategory.ADJACENT_CONCEPT: "adjacent_concepts",
            SemanticCategory.BUSINESS_GOAL: "business_goals",
            SemanticCategory.USER_INTENT: "user_intents",
            SemanticCategory.NEGATIVE_CONTEXT: "negative_contexts",
        }
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
        detected_domain=data.get("detected_domain", "general"),
        weighted_terms=weighted,
    )


class SWAGProfileGenerator:
    """Generates SWAG profiles for chunks and queries via LLM."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        config: Optional[SWAGConfig] = None,
    ):
        self.config = config or SWAGConfig.from_env()
        self.llm = llm_client or create_llm_client(self.config)

    def generate_chunk_profile(
        self,
        chunk_text: str,
        source_filename: str = "unknown",
        page_number: int | None = None,
    ) -> SWAGProfile:
        """
        Generate SWAG metadata for a document chunk.
        Metadata is stored separately — never appended to chunk_text.
        """
        user_prompt = SWAG_CHUNK_USER_TEMPLATE.format(
            chunk_text=chunk_text,
            source_filename=source_filename,
            page_number=page_number or "N/A",
        )

        try:
            data = self.llm.complete_json(SWAG_SYSTEM_PROMPT, user_prompt)
            profile = _build_profile_from_json(data)
            logger.debug(
                "Generated SWAG profile for %s (domain=%s, confidence=%.2f)",
                source_filename,
                profile.detected_domain,
                profile.confidence,
            )
            return profile
        except Exception as exc:
            logger.warning("SWAG profile generation failed: %s — using minimal profile", exc)
            return SWAGProfile(confidence=0.0, detected_domain="unknown")

    def generate_query_profile(self, query: str) -> SWAGProfile:
        """Generate SWAG metadata for a user query."""
        user_prompt = SWAG_QUERY_USER_TEMPLATE.format(query=query)

        try:
            data = self.llm.complete_json(SWAG_SYSTEM_PROMPT, user_prompt)
            return _build_profile_from_json(data)
        except Exception as exc:
            logger.warning("Query profile generation failed: %s", exc)
            return SWAGProfile(
                primary_concepts=[query],
                confidence=0.3,
                detected_domain="general",
            )
