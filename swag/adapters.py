"""Convert between DocumentChunk models and dict hits used by the SWAG reranker."""

from __future__ import annotations

from swag.schemas import DocumentChunk, RetrievalResult


def chunk_to_hit(chunk: DocumentChunk, vector_score: float) -> dict:
    doc_id = str(chunk.metadata.get("document_id") or chunk.source_filename)
    chunk_index = int(chunk.metadata.get("chunk_index", 0))
    seed_key = chunk.metadata.get("seed_key")
    chunk_kind = chunk.metadata.get("chunk_kind")
    return {
        "id": chunk.chunk_id,
        "document_id": doc_id,
        "chunk_index": chunk_index,
        "text": chunk.raw_text,
        "source": chunk.source_filename,
        "page": chunk.page_number,
        "score": vector_score,
        "swag_profile": chunk.swag_profile.model_dump() if chunk.swag_profile else None,
        "seed_key": seed_key,
        "topic": chunk.metadata.get("topic"),
        "chunk_kind": chunk_kind,
        "_chunk": chunk,
    }


def hit_to_retrieval_result(hit: dict, rank: int) -> RetrievalResult:
    chunk: DocumentChunk | None = hit.get("_chunk")
    return RetrievalResult(
        chunk_id=str(hit.get("id") or (chunk.chunk_id if chunk else "")),
        raw_text=str(hit.get("text") or (chunk.raw_text if chunk else "")),
        source_document=str(hit.get("source") or (chunk.source_filename if chunk else "")),
        page_number=hit.get("page")
        if hit.get("page") is not None
        else (chunk.page_number if chunk else None),
        vector_similarity=float(hit.get("vector_score") or 0.0),
        swag_score=float(hit.get("swag_score") or 0.0),
        metadata_boost=float(hit.get("metadata_boost") or 0.0),
        final_score=float(hit.get("score") or 0.0),
        explanation=str(hit.get("swag_explanation") or hit.get("explanation") or ""),
        rank=rank,
        chunk_kind=str(hit.get("chunk_kind") or "body"),
    )
