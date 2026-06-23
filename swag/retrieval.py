"""Retrieval pipeline — baseline RAG + SWAG reranking."""

from __future__ import annotations

import logging
from typing import Optional

from swag.adapters import chunk_to_hit, hit_to_retrieval_result
from swag.config import SWAGConfig
from swag.embeddings import BaseEmbedder, create_embedder
from swag.llm_client import BaseLLMClient, create_llm_client
from swag.profile import SWAGProfileGenerator
from swag.rerank import rerank_hits_with_swag
from swag.schemas import AnswerResult, RetrievalResult
from swag.vector_store import BaseVectorStore, SearchResult, create_vector_store

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Embed → vector search → SWAG rerank → optional answer generation."""

    def __init__(
        self,
        config: Optional[SWAGConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        profile_generator: Optional[SWAGProfileGenerator] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or SWAGConfig.from_env()
        self.embedder = embedder or create_embedder(self.config)
        self.vector_store = vector_store or create_vector_store(
            self.config, dimension=self.embedder.dimension
        )
        self.profile_generator = profile_generator or SWAGProfileGenerator()
        self.llm = llm_client or create_llm_client(self.config)

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def retrieve_baseline(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self.config.top_k
        results = self._vector_search(query, k)
        return [
            RetrievalResult(
                chunk_id=r.chunk.chunk_id,
                raw_text=r.chunk.raw_text,
                source_document=r.chunk.source_filename,
                page_number=r.chunk.page_number,
                vector_similarity=r.score,
                swag_score=0.0,
                metadata_boost=0.0,
                final_score=r.score,
                explanation=f"Baseline vector similarity: {r.score:.3f}",
                rank=i + 1,
            )
            for i, r in enumerate(results)
        ]

    def retrieve_with_swag(
        self, query: str, top_k: Optional[int] = None
    ) -> tuple[list[RetrievalResult], dict]:
        k = top_k or self.config.top_k
        candidate_k = max(self.config.retrieval_candidates, k)
        candidates_raw = self._vector_search(query, candidate_k)
        hits = [chunk_to_hit(sr.chunk, sr.score) for sr in candidates_raw]

        reranked, swag_meta = rerank_hits_with_swag(
            query,
            hits,
            collection=self.config.vector_store.collection_name,
            top_k=k,
            vector_store=self.vector_store,
        )

        results = [hit_to_retrieval_result(hit, i + 1) for i, hit in enumerate(reranked)]
        return results, swag_meta

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        generate_answer: bool = True,
        use_swag: bool = True,
    ) -> AnswerResult:
        baseline = self.retrieve_baseline(question, top_k)
        swag_meta: dict = {}
        if use_swag and self.config.swag.swag_enabled:
            swag_results, swag_meta = self.retrieve_with_swag(question, top_k)
        else:
            swag_results = baseline

        answer = ""
        if generate_answer:
            context_chunks = [r.raw_text for r in swag_results[:3]]
            if context_chunks:
                answer = self.llm.generate_answer(question, context_chunks)
            else:
                answer = "No relevant documents found to answer this question."

        return AnswerResult(
            query=question,
            answer=answer,
            baseline_results=baseline,
            swag_results=swag_results,
            used_swag=use_swag and self.config.swag.swag_enabled,
            swag_meta=swag_meta,
        )
