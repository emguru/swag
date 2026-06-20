"""Embedding generation — sentence-transformers (local) or OpenAI."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from swag.config import EmbeddingConfig, SWAGConfig

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformers model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=len(texts) > 10)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding API."""

    def __init__(self, model_name: str, api_key: str, dimension: int = 1536):
        self.model_name = model_name
        self.api_key = api_key
        self._dimension = dimension
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 100
        all_vectors: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            all_vectors.extend([item.embedding for item in response.data])
        return all_vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def create_embedder(swag_config: Optional[SWAGConfig] = None) -> BaseEmbedder:
    """Factory: create embedder based on config."""
    cfg = swag_config or SWAGConfig.from_env()
    provider = cfg.embedding.provider.lower()

    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.embedding.model_name)
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        return OpenAIEmbedder(
            cfg.embedding.openai_model,
            cfg.openai_api_key,
            cfg.embedding.dimension,
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider}. "
        "Supported: sentence-transformers, openai."
    )
