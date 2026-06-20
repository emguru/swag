"""
FastAPI demo application for SWAG RAG.

Run: uvicorn swag.app:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from swag.config import SWAGConfig
from swag.ingest import IngestionPipeline
from swag.retrieval import RetrievalPipeline
from swag.schemas import AnswerResult, IngestResult, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SWAG RAG",
    description="Semantic Weighted Augmented Generation — retrieval enhancement for RAG",
    version="0.1.0",
)

# Shared pipeline state (initialized on startup)
_config: SWAGConfig | None = None
_ingestion: IngestionPipeline | None = None
_retrieval: RetrievalPipeline | None = None


def get_pipelines() -> tuple[IngestionPipeline, RetrievalPipeline]:
    global _config, _ingestion, _retrieval
    if _ingestion is None or _retrieval is None:
        _config = SWAGConfig.from_env()
        _config.ensure_dirs()
        _ingestion = IngestionPipeline(config=_config)
        _retrieval = RetrievalPipeline(
            config=_config,
            embedder=_ingestion.embedder,
            vector_store=_ingestion.vector_store,
            profile_generator=_ingestion.profile_generator,
        )
        # Try loading existing index
        if _ingestion.load_index():
            logger.info("Loaded existing index with %d chunks", _ingestion.vector_store.count())
    return _ingestion, _retrieval


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="Directory to ingest (default: data/)")
    generate_swag: bool = Field(default=True)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    generate_answer: bool = Field(default=True)
    use_swag: bool = Field(default=True)


class CompareResponse(BaseModel):
    query: str
    answer: str
    baseline: list[RetrievalResult]
    swag: list[RetrievalResult]


@app.on_event("startup")
async def startup():
    get_pipelines()


@app.get("/")
async def root():
    ingestion, retrieval = get_pipelines()
    return {
        "name": "SWAG RAG",
        "version": "0.1.0",
        "indexed_chunks": ingestion.vector_store.count(),
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "POST /query",
            "compare": "POST /compare",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    ingestion, _ = get_pipelines()
    return {"status": "ok", "indexed_chunks": ingestion.vector_store.count()}


@app.post("/ingest", response_model=list[IngestResult])
async def ingest_documents(request: IngestRequest):
    """Ingest documents from data directory, build index, generate SWAG profiles."""
    ingestion, _ = get_pipelines()
    directory = Path(request.directory) if request.directory else ingestion.config.data_dir

    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    results = ingestion.ingest_directory(directory, generate_swag=request.generate_swag)
    ingestion.save_index()
    return results


@app.post("/query", response_model=AnswerResult)
async def query_documents(request: QueryRequest):
    """Query with SWAG reranking and optional answer generation."""
    _, retrieval = get_pipelines()

    if retrieval.vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. POST /ingest first.",
        )

    return retrieval.query(
        question=request.question,
        top_k=request.top_k,
        generate_answer=request.generate_answer,
        use_swag=request.use_swag,
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_retrieval(request: QueryRequest):
    """Compare baseline RAG vs SWAG-reranked results side by side."""
    _, retrieval = get_pipelines()

    if retrieval.vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. POST /ingest first.")

    result = retrieval.query(
        question=request.question,
        top_k=request.top_k,
        generate_answer=request.generate_answer,
        use_swag=True,
    )

    return CompareResponse(
        query=result.query,
        answer=result.answer,
        baseline=result.baseline_results,
        swag=result.swag_results,
    )
