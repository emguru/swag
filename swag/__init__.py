"""SWAG — Semantic Weighted Augmented Generation for RAG."""

__version__ = "0.1.0"

from swag.config import SWAGConfig, config
from swag.ingest import IngestionPipeline
from swag.retrieval import RetrievalPipeline
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
    "DocumentChunk",
    "SWAGProfile",
    "WeightedSemanticTerm",
    "QueryProfile",
    "RetrievalResult",
    "AnswerResult",
]
