"""Modular vector store — FAISS default, extensible to Qdrant/Pinecone/Chroma."""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from swag.config import VectorStoreConfig, SWAGConfig
from swag.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class SearchResult:
    """Internal search result from vector store."""

    def __init__(self, chunk: DocumentChunk, score: float):
        self.chunk = chunk
        self.score = score


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Add chunks with their embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """Search for similar chunks."""

    @abstractmethod
    def save(self) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self) -> bool:
        """Load index from disk. Returns True if loaded successfully."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed chunks."""

    def update_chunk_profile(self, chunk_id: str, profile) -> None:
        """Persist an updated SWAG profile for a chunk (optional backend support)."""
        del chunk_id, profile

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all stored chunks (for SWAG profile lookup)."""
        return []


class FAISSVectorStore(BaseVectorStore):
    """Local FAISS vector index with metadata sidecar."""

    def __init__(self, config: VectorStoreConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        self._index = None
        self._chunks: list[DocumentChunk] = []
        self._id_map: dict[int, str] = {}  # faiss idx -> chunk_id

        self.index_path = Path(config.index_path)
        self.metadata_path = self.index_path.with_suffix(".meta.pkl")
        self.swag_path = self.index_path.with_suffix(".swag.json")

    @property
    def index(self):
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        vectors = np.array(embeddings, dtype=np.float32)
        # Normalize for inner product = cosine similarity
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(vectors)

        start_idx = len(self._chunks)
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._id_map[start_idx + i] = chunk.chunk_id

        logger.info("Added %d chunks to FAISS index (total: %d)", len(chunks), len(self._chunks))

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if not self._chunks:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss_module = __import__("faiss")
        faiss_module.normalize_L2(query_vec)

        k = min(top_k, len(self._chunks))
        scores, indices = self.index.search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            # Inner product of normalized vectors is in [-1, 1]; map to [0, 1]
            normalized_score = (float(score) + 1.0) / 2.0
            results.append(SearchResult(chunk=chunk, score=normalized_score))

        return results

    def save(self) -> None:
        import faiss

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with open(self.metadata_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "id_map": self._id_map}, f)

        # Persist SWAG profiles separately
        swag_data = {
            c.chunk_id: c.swag_profile.model_dump() if c.swag_profile else None
            for c in self._chunks
        }
        with open(self.swag_path, "w", encoding="utf-8") as f:
            json.dump(swag_data, f, indent=2, default=str)

        logger.info("Saved FAISS index to %s", self.index_path)

    def load(self) -> bool:
        import faiss

        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)
            self._chunks = data["chunks"]
            self._id_map = data.get("id_map", {})

        # Reload SWAG profiles
        if self.swag_path.exists():
            from swag.schemas import SWAGProfile

            with open(self.swag_path, encoding="utf-8") as f:
                swag_data = json.load(f)
            for chunk in self._chunks:
                profile_data = swag_data.get(chunk.chunk_id)
                if profile_data:
                    chunk.swag_profile = SWAGProfile(**profile_data)

        logger.info("Loaded FAISS index with %d chunks", len(self._chunks))
        return True

    def count(self) -> int:
        return len(self._chunks)

    def get_all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def update_chunk_profile(self, chunk_id: str, profile) -> None:
        from swag.schemas import SWAGProfile

        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                chunk.swag_profile = profile if isinstance(profile, SWAGProfile) else SWAGProfile(**profile)
                return


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma vector store stub — implement for production use.

    To enable: pip install chromadb
    Set VECTOR_STORE_BACKEND=chroma in .env
    """

    def __init__(self, config: VectorStoreConfig, dimension: int):
        raise NotImplementedError(
            "Chroma backend stub. Install chromadb and implement ChromaVectorStore "
            "following the FAISSVectorStore pattern."
        )

    def add(self, chunks, embeddings):
        pass

    def search(self, query_embedding, top_k):
        return []

    def save(self):
        pass

    def load(self):
        return False

    def count(self):
        return 0


class QdrantVectorStore(BaseVectorStore):
    """Qdrant backend for on-prem or cloud vector search."""

    def __init__(self, config: VectorStoreConfig, dimension: int):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as exc:
            raise ImportError("Install qdrant-client: pip install qdrant-client") from exc

        self.config = config
        self.dimension = dimension
        self._PointStruct = PointStruct
        self.client = QdrantClient(url=config.qdrant_url)
        self.collection = config.collection_name
        self._chunks_cache: dict[str, DocumentChunk] = {}

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def _point_id(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _chunk_payload(self, chunk: DocumentChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source_filename": chunk.source_filename,
            "page_number": chunk.page_number,
            "raw_text": chunk.raw_text,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "metadata": chunk.metadata,
            "swag_profile": chunk.swag_profile.model_dump() if chunk.swag_profile else None,
        }

    def _chunk_from_payload(self, payload: dict) -> DocumentChunk:
        from swag.schemas import SWAGProfile

        profile_data = payload.get("swag_profile")
        profile = SWAGProfile(**profile_data) if profile_data else None
        return DocumentChunk(
            chunk_id=payload["chunk_id"],
            source_filename=payload.get("source_filename", "unknown"),
            page_number=payload.get("page_number"),
            raw_text=payload.get("raw_text", ""),
            char_start=int(payload.get("char_start", 0)),
            char_end=int(payload.get("char_end", 0)),
            metadata=payload.get("metadata") or {},
            swag_profile=profile,
        )

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        points = []
        for chunk, vector in zip(chunks, embeddings):
            self._chunks_cache[chunk.chunk_id] = chunk
            points.append(
                self._PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload=self._chunk_payload(chunk),
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Upserted %d chunks to Qdrant collection %s", len(points), self.collection)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k,
        )
        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = self._chunk_from_payload(payload)
            self._chunks_cache[chunk.chunk_id] = chunk
            score = float(hit.score)
            normalized = score if 0.0 <= score <= 1.0 else max(0.0, min(1.0, (score + 1.0) / 2.0))
            results.append(SearchResult(chunk=chunk, score=normalized))
        return results

    def save(self) -> None:
        return None

    def load(self) -> bool:
        return self.count() > 0

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def update_chunk_profile(self, chunk_id: str, profile) -> None:
        from swag.schemas import SWAGProfile

        prof = profile if isinstance(profile, SWAGProfile) else SWAGProfile(**profile)
        self.client.set_payload(
            collection_name=self.collection,
            payload={"swag_profile": prof.model_dump()},
            points=[self._point_id(chunk_id)],
        )
        if chunk_id in self._chunks_cache:
            self._chunks_cache[chunk_id].swag_profile = prof


def create_vector_store(
    swag_config: Optional[SWAGConfig] = None,
    dimension: Optional[int] = None,
) -> BaseVectorStore:
    """Factory: create vector store based on config."""
    cfg = swag_config or SWAGConfig.from_env()
    backend = cfg.vector_store.backend.lower()
    dim = dimension or cfg.embedding.dimension

    stores = {
        "faiss": FAISSVectorStore,
        "chroma": ChromaVectorStore,
        "qdrant": QdrantVectorStore,
    }

    if backend not in stores:
        raise ValueError(
            f"Unsupported vector store: {backend}. Supported: {list(stores.keys())}"
        )

    return stores[backend](cfg.vector_store, dim)
