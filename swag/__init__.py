"""SWAG — Semantic Weighted Augmented Generation for RAG."""

__version__ = "0.3.0"

from swag.config import SWAGConfig, config
from swag.ingest import IngestionPipeline
from swag.profile import SWAGProfileGenerator
from swag.retrieval import RetrievalPipeline
from swag.rerank import rerank_hits_with_swag
from swag.schemas import (
    AnswerResult,
    DocumentChunk,
    QueryProfile,
    RetrievalResult,
    SWAGProfile,
    WeightedSemanticTerm,
)

__all__ = [
    "SWAGConfig",
    "config",
    "IngestionPipeline",
    "RetrievalPipeline",
    "SWAGProfileGenerator",
    "rerank_hits_with_swag",
    "DocumentChunk",
    "SWAGProfile",
    "WeightedSemanticTerm",
    "QueryProfile",
    "RetrievalResult",
    "AnswerResult",
]
