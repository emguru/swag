"""Configuration for the SWAG RAG framework."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=512, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=64, description="Overlap between consecutive chunks")


class ScoringConfig(BaseModel):
    vector_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    swag_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    metadata_weight: float = Field(default=0.05, ge=0.0, le=1.0)

    def validate_weights(self) -> None:
        total = self.vector_weight + self.swag_weight + self.metadata_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0 (got {total:.3f}). "
                "Adjust VECTOR_WEIGHT, SWAG_WEIGHT, METADATA_WEIGHT."
            )


class EmbeddingConfig(BaseModel):
    provider: str = Field(default="sentence-transformers")  # or "openai"
    model_name: str = Field(default="all-MiniLM-L6-v2")
    openai_model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=384)


class LLMConfig(BaseModel):
    provider: str = Field(default="openai")  # extensible: anthropic, ollama, etc.
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=2048)


class VectorStoreConfig(BaseModel):
    backend: str = Field(default="faiss")  # faiss | qdrant | pinecone | chroma
    index_path: Path = Field(default=_PROJECT_ROOT / "indexes" / "faiss_index")
    collection_name: str = Field(default="swag_chunks")


class SWAGConfig(BaseModel):
    """Top-level configuration."""

    project_root: Path = Field(default=_PROJECT_ROOT)
    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    indexes_dir: Path = Field(default=_PROJECT_ROOT / "indexes")

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    # Retrieval defaults
    top_k: int = Field(default=10)
    retrieval_candidates: int = Field(default=20)

    # API keys from environment
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    @classmethod
    def from_env(cls) -> "SWAGConfig":
        """Build config from environment variables with sensible defaults."""
        scoring = ScoringConfig(
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.65")),
            swag_weight=float(os.getenv("SWAG_WEIGHT", "0.30")),
            metadata_weight=float(os.getenv("METADATA_WEIGHT", "0.05")),
        )
        scoring.validate_weights()

        return cls(
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            ),
            scoring=scoring,
            embedding=EmbeddingConfig(
                provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            ),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            ),
            vector_store=VectorStoreConfig(
                backend=os.getenv("VECTOR_STORE_BACKEND", "faiss"),
            ),
            top_k=int(os.getenv("TOP_K", "10")),
            retrieval_candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
        )

    def ensure_dirs(self) -> None:
        """Create required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.index_path.parent.mkdir(parents=True, exist_ok=True)


# Global default config instance
config = SWAGConfig.from_env()
