"""Retrieval pipeline — baseline RAG + SWAG reranking."""

from __future__ import annotations

import logging
from typing import Optional

from swag.config import SWAGConfig
from swag.embeddings import BaseEmbedder, create_embedder
from swag.llm_client import BaseLLMClient, create_llm_client
from swag.schemas import (
    AnswerResult,
    QueryProfile,
    RetrievalCandidate,
    RetrievalResult,
)
from swag.swag_profile import SWAGProfileGenerator
from swag.swag_scoring import (
    build_selection_explanation,
    compute_final_score,
    compute_metadata_boost,
    score_swag_alignment,
)
from swag.vector_store import BaseVectorStore, SearchResult, create_vector_store

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Full retrieval pipeline:
    1. Embed query
    2. Vector search for candidates
    3. Generate query SWAG profile
    4. Score and rerank with SWAG
    5. Optionally generate answer
    """

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
        self.profile_generator = profile_generator or SWAGProfileGenerator(config=self.config)
        self.llm = llm_client or create_llm_client(self.config)

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Baseline vector similarity search."""
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def retrieve_baseline(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """Retrieve using vector similarity only (traditional RAG)."""
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

    def retrieve_with_swag(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """
        Retrieve with SWAG reranking:
        1. Get candidate pool from vector search
        2. Generate query SWAG profile
        3. Score each candidate's SWAG profile
        4. Combine scores and rerank
        """
        k = top_k or self.config.top_k
        candidate_k = max(self.config.retrieval_candidates, k)
        candidates_raw = self._vector_search(query, candidate_k)

        # Generate query semantic profile
        query_swag = self.profile_generator.generate_query_profile(query)
        query_profile = QueryProfile(original_query=query, swag_profile=query_swag)

        scoring = self.config.scoring
        candidates: list[RetrievalCandidate] = []

        for search_result in candidates_raw:
            chunk = search_result.chunk
            vector_sim = search_result.score

            chunk_profile = chunk.swag_profile
            if chunk_profile:
                swag_score, swag_explanations = score_swag_alignment(
                    query_profile.swag_profile, chunk_profile
                )
            else:
                swag_score, swag_explanations = 0.0, ["No SWAG profile on chunk"]

            metadata_boost = compute_metadata_boost(
                chunk, query_profile.swag_profile, original_query=query
            )

            final = compute_final_score(
                vector_sim,
                swag_score,
                metadata_boost,
                scoring.vector_weight,
                scoring.swag_weight,
                scoring.metadata_weight,
            )

            explanation = build_selection_explanation(
                vector_sim, swag_score, metadata_boost, final, swag_explanations
            )

            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    vector_similarity=vector_sim,
                    swag_score=swag_score,
                    metadata_boost=metadata_boost,
                    final_score=final,
                    explanation=explanation,
                )
            )

        # Rerank by final score
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        top_candidates = candidates[:k]

        return [
            RetrievalResult(
                chunk_id=c.chunk.chunk_id,
                raw_text=c.chunk.raw_text,
                source_document=c.chunk.source_filename,
                page_number=c.chunk.page_number,
                vector_similarity=c.vector_similarity,
                swag_score=c.swag_score,
                metadata_boost=c.metadata_boost,
                final_score=c.final_score,
                explanation=c.explanation,
                rank=i + 1,
            )
            for i, c in enumerate(top_candidates)
        ]

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        generate_answer: bool = True,
        use_swag: bool = True,
    ) -> AnswerResult:
        """
        Full RAG query: retrieve + optional answer generation.
        Returns both baseline and SWAG results for comparison.
        """
        baseline = self.retrieve_baseline(question, top_k)
        swag_results = self.retrieve_with_swag(question, top_k) if use_swag else baseline

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
            used_swag=use_swag,
        )
