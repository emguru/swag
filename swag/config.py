"""Configuration for the SWAG RAG framework — local, on-prem, or cloud."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class SwagTuningConfig(BaseModel):
    """Reranking engine knobs (ported from kube_rag_langg)."""

    swag_enabled: bool = True
    swag_ingest_profiles: bool = True
    swag_profile_mode: str = "heuristic"  # heuristic | llm
    swag_grader_model: str = "gpt-4o-mini"
    swag_min_score_threshold: float = 0.08
    swag_noise_vector_margin: float = 0.04
    swag_noise_relative_vector_ratio: float = 0.65
    swag_noise_backfill_ratio: float = 0.58
    swag_evaluative_blend_boost: float = 0.15
    swag_evaluative_title_penalty: float = 0.65
    swag_chunk_title_multiplier: float = 0.40
    swag_chunk_nav_multiplier: float = 0.30
    swag_chunk_boilerplate_multiplier: float = 0.02
    swag_intent_category_boost: float = 1.25
    swag_max_chunks_per_source: int = 3
    swag_deduplicate_similarity: float = 0.92
    swag_adaptive_blend: bool = True
    swag_adaptive_vector_epsilon: float = 0.03
    swag_adaptive_swag_boost: float = 0.12
    swag_persist_profiles: bool = True
    swag_max_terms_per_list: int = 12
    swag_max_weighted_terms: int = 24
    swag_max_term_chars: int = 80
    swag_max_reason_chars: int = 120
    swag_max_domain_chars: int = 64
    swag_ungrounded_term_weight_multiplier: float = 0.35
    swag_grounding_min_token_ratio: float = 0.5
    swag_fuzzy_min_token_len: int = 4


class EmbeddingConfig(BaseModel):
    provider: str = Field(default="sentence-transformers")  # sentence-transformers | openai
    model_name: str = Field(default="all-MiniLM-L6-v2")
    openai_model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=384)


class LLMConfig(BaseModel):
    provider: str = Field(default="openai")  # openai | ollama | mock
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=2048)
    ollama_base_url: str = Field(default="http://localhost:11434")


class VectorStoreConfig(BaseModel):
    backend: str = Field(default="faiss")  # faiss | qdrant
    index_path: Path = Field(default=_PROJECT_ROOT / "indexes" / "faiss_index")
    collection_name: str = Field(default="swag_chunks")
    qdrant_url: str = Field(default="http://localhost:6333")


class SWAGConfig(BaseModel):
    """Top-level configuration."""

    project_root: Path = Field(default=_PROJECT_ROOT)
    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    indexes_dir: Path = Field(default=_PROJECT_ROOT / "indexes")

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    swag: SwagTuningConfig = Field(default_factory=SwagTuningConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    top_k: int = Field(default=10)
    retrieval_candidates: int = Field(default=20)
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)

    @classmethod
    def from_env(cls) -> "SWAGConfig":
        scoring = ScoringConfig(
            vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.65")),
            swag_weight=float(os.getenv("SWAG_WEIGHT", "0.30")),
            metadata_weight=float(os.getenv("METADATA_WEIGHT", "0.05")),
        )
        scoring.validate_weights()

        swag = SwagTuningConfig(
            swag_enabled=os.getenv("SWAG_ENABLED", "true").lower() in {"1", "true", "yes"},
            swag_ingest_profiles=os.getenv("SWAG_INGEST_PROFILES", "true").lower()
            in {"1", "true", "yes"},
            swag_profile_mode=os.getenv("SWAG_PROFILE_MODE", "heuristic"),
            swag_grader_model=os.getenv("SWAG_GRADER_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
            swag_min_score_threshold=float(os.getenv("SWAG_MIN_SCORE_THRESHOLD", "0.08")),
            swag_noise_vector_margin=float(os.getenv("SWAG_NOISE_VECTOR_MARGIN", "0.04")),
            swag_noise_relative_vector_ratio=float(
                os.getenv("SWAG_NOISE_RELATIVE_VECTOR_RATIO", "0.65")
            ),
            swag_noise_backfill_ratio=float(os.getenv("SWAG_NOISE_BACKFILL_RATIO", "0.58")),
            swag_evaluative_blend_boost=float(os.getenv("SWAG_EVALUATIVE_BLEND_BOOST", "0.15")),
            swag_evaluative_title_penalty=float(os.getenv("SWAG_EVALUATIVE_TITLE_PENALTY", "0.65")),
            swag_chunk_title_multiplier=float(os.getenv("SWAG_CHUNK_TITLE_MULTIPLIER", "0.40")),
            swag_chunk_nav_multiplier=float(os.getenv("SWAG_CHUNK_NAV_MULTIPLIER", "0.30")),
            swag_chunk_boilerplate_multiplier=float(
                os.getenv("SWAG_CHUNK_BOILERPLATE_MULTIPLIER", "0.02")
            ),
            swag_intent_category_boost=float(os.getenv("SWAG_INTENT_CATEGORY_BOOST", "1.25")),
            swag_max_chunks_per_source=int(os.getenv("SWAG_MAX_CHUNKS_PER_SOURCE", "3")),
            swag_deduplicate_similarity=float(os.getenv("SWAG_DEDUPLICATE_SIMILARITY", "0.92")),
            swag_adaptive_blend=os.getenv("SWAG_ADAPTIVE_BLEND", "true").lower()
            in {"1", "true", "yes"},
            swag_adaptive_vector_epsilon=float(os.getenv("SWAG_ADAPTIVE_VECTOR_EPSILON", "0.03")),
            swag_adaptive_swag_boost=float(os.getenv("SWAG_ADAPTIVE_SWAG_BOOST", "0.12")),
            swag_persist_profiles=os.getenv("SWAG_PERSIST_PROFILES", "true").lower()
            in {"1", "true", "yes"},
            swag_max_terms_per_list=int(os.getenv("SWAG_MAX_TERMS_PER_LIST", "12")),
            swag_max_weighted_terms=int(os.getenv("SWAG_MAX_WEIGHTED_TERMS", "24")),
            swag_max_term_chars=int(os.getenv("SWAG_MAX_TERM_CHARS", "80")),
            swag_max_reason_chars=int(os.getenv("SWAG_MAX_REASON_CHARS", "120")),
            swag_max_domain_chars=int(os.getenv("SWAG_MAX_DOMAIN_CHARS", "64")),
            swag_ungrounded_term_weight_multiplier=float(
                os.getenv("SWAG_UNGROUNDED_TERM_WEIGHT_MULTIPLIER", "0.35")
            ),
            swag_grounding_min_token_ratio=float(
                os.getenv("SWAG_GROUNDING_MIN_TOKEN_RATIO", "0.5")
            ),
            swag_fuzzy_min_token_len=int(os.getenv("SWAG_FUZZY_MIN_TOKEN_LEN", "4")),
        )

        data_dir = Path(os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data")))
        indexes_dir = Path(os.getenv("INDEXES_DIR", str(_PROJECT_ROOT / "indexes")))

        return cls(
            project_root=_PROJECT_ROOT,
            data_dir=data_dir,
            indexes_dir=indexes_dir,
            chunking=ChunkingConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            ),
            scoring=scoring,
            swag=swag,
            embedding=EmbeddingConfig(
                provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
                model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            ),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            vector_store=VectorStoreConfig(
                backend=os.getenv("VECTOR_STORE_BACKEND", "faiss"),
                index_path=indexes_dir / "faiss_index",
                collection_name=os.getenv("QDRANT_COLLECTION", "swag_chunks"),
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            ),
            top_k=int(os.getenv("TOP_K", "10")),
            retrieval_candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8080")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.index_path.parent.mkdir(parents=True, exist_ok=True)


config = SWAGConfig.from_env()
