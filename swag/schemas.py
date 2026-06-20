"""Pydantic schemas for the SWAG RAG framework."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SemanticCategory(str, Enum):
    """Categories for weighted semantic terms."""

    PRIMARY_CONCEPT = "primary_concept"
    DOMAIN_TERM = "domain_term"
    DIRECT_SYNONYM = "direct_synonym"
    ADJACENT_CONCEPT = "adjacent_concept"
    BUSINESS_GOAL = "business_goal"
    USER_INTENT = "user_intent"
    NEGATIVE_CONTEXT = "negative_context"


class WeightedSemanticTerm(BaseModel):
    """A single weighted semantic relationship tied to a chunk."""

    term: str = Field(..., description="The semantic term or phrase")
    category: SemanticCategory = Field(..., description="Category of the relationship")
    weight: float = Field(..., ge=0.0, le=1.0, description="Contextual relevance weight")
    reason: str = Field(default="", description="Why this term relates to the chunk")

    @field_validator("term")
    @classmethod
    def term_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("term must not be empty")
        return v.strip()


class SWAGProfile(BaseModel):
    """
    Structured semantic profile for a chunk or query.

    IMPORTANT: This metadata is for reranking only — never appended to chunk text
    and never treated as source-of-truth facts.
    """

    primary_concepts: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    direct_synonyms: list[str] = Field(default_factory=list)
    adjacent_concepts: list[str] = Field(default_factory=list)
    business_goals: list[str] = Field(default_factory=list)
    user_intents: list[str] = Field(default_factory=list)
    negative_contexts: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detected_domain: str = Field(default="general", description="Detected domain context")
    weighted_terms: list[WeightedSemanticTerm] = Field(default_factory=list)

    def to_category_lists(self) -> dict[str, list[str]]:
        """Return category -> terms mapping for scoring."""
        return {
            "primary_concepts": self.primary_concepts,
            "domain_terms": self.domain_terms,
            "direct_synonyms": self.direct_synonyms,
            "adjacent_concepts": self.adjacent_concepts,
            "business_goals": self.business_goals,
            "user_intents": self.user_intents,
            "negative_contexts": self.negative_contexts,
        }


class DocumentChunk(BaseModel):
    """An ingested document chunk — original text remains authoritative."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    source_filename: str = Field(..., description="Original source file name")
    page_number: Optional[int] = Field(default=None, description="Page number if available")
    raw_text: str = Field(..., description="Original chunk text — authoritative source")
    char_start: int = Field(default=0, description="Character offset in source document")
    char_end: int = Field(default=0, description="End character offset in source document")
    swag_profile: Optional[SWAGProfile] = Field(
        default=None, description="SWAG metadata stored separately from raw_text"
    )
    embedding: Optional[list[float]] = Field(default=None, exclude=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryProfile(BaseModel):
    """Semantic profile extracted from a user query."""

    original_query: str
    swag_profile: SWAGProfile
    embedding: Optional[list[float]] = Field(default=None, exclude=True)


class RetrievalCandidate(BaseModel):
    """A candidate chunk from vector search before reranking."""

    chunk: DocumentChunk
    vector_similarity: float = Field(..., ge=0.0, le=1.0)
    swag_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_boost: float = Field(default=0.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="")


class RetrievalResult(BaseModel):
    """Final reranked retrieval result returned to the caller."""

    chunk_id: str
    raw_text: str
    source_document: str
    page_number: Optional[int] = None
    vector_similarity: float
    swag_score: float
    metadata_boost: float
    final_score: float
    explanation: str
    rank: int = 0


class IngestResult(BaseModel):
    """Summary of an ingestion run."""

    source_filename: str
    num_chunks: int
    chunk_ids: list[str]


class AnswerResult(BaseModel):
    """Full RAG answer with retrieval details."""

    query: str
    answer: str
    baseline_results: list[RetrievalResult]
    swag_results: list[RetrievalResult]
    used_swag: bool = True
