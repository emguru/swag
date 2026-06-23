"""
FastAPI service for SWAG RAG — local, on-prem, or cloud.

Run locally:
  uvicorn swag.app:app --reload --port 8080

Docker:
  docker compose up --build
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from swag import __version__
from swag.config import SWAGConfig
from swag.ingest import IngestionPipeline
from swag.retrieval import RetrievalPipeline
from swag.schemas import AnswerResult, IngestResult, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SWAG RAG",
    description="Semantic Weighted Augmented Generation — retrieval enhancement for RAG",
    version=__version__,
)

_config: SWAGConfig | None = None
_ingestion: IngestionPipeline | None = None
_retrieval: RetrievalPipeline | None = None


def get_pipelines() -> tuple[IngestionPipeline, RetrievalPipeline]:
    global _config, _ingestion, _retrieval
    if _ingestion is None or _retrieval is None:
        _config = SWAGConfig.from_env()
        _config.ensure_dirs()
        from swag.settings import reload_settings

        reload_settings(_config)
        _ingestion = IngestionPipeline(config=_config)
        _retrieval = RetrievalPipeline(
            config=_config,
            embedder=_ingestion.embedder,
            vector_store=_ingestion.vector_store,
            profile_generator=_ingestion.profile_generator,
        )
        if _ingestion.load_index():
            logger.info("Loaded index with %d chunks", _ingestion.vector_store.count())
    return _ingestion, _retrieval


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="Directory to ingest (default: data/)")
    generate_swag: bool = Field(default=True)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    generate_answer: bool = Field(default=True)
    use_swag: bool = Field(default=True)
    include_text: bool = Field(
        default=False,
        description="Include full chunk text in results (default: scores and source only)",
    )
    include_explanation: bool = Field(
        default=False,
        description="Include per-hit SWAG scoring explanations",
    )


class CompareResponse(BaseModel):
    query: str
    answer: str
    baseline: list[RetrievalResult]
    swag: list[RetrievalResult]
    swag_meta: dict | None = None


def _trim_results(
    results: list[RetrievalResult],
    *,
    include_text: bool,
    include_explanation: bool,
) -> list[RetrievalResult]:
    if include_text and include_explanation:
        return results
    trimmed: list[RetrievalResult] = []
    for r in results:
        trimmed.append(
            r.model_copy(
                update={
                    "raw_text": r.raw_text if include_text else "",
                    "explanation": r.explanation if include_explanation else "",
                }
            )
        )
    return trimmed


@app.on_event("startup")
async def startup():
    get_pipelines()


@app.get("/")
async def root():
    ingestion, _ = get_pipelines()
    cfg = ingestion.config
    return {
        "name": "SWAG RAG",
        "version": __version__,
        "indexed_chunks": ingestion.vector_store.count(),
        "vector_store": cfg.vector_store.backend,
        "swag_profile_mode": cfg.swag.swag_profile_mode,
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "POST /query",
            "compare": "POST /compare",
            "health": "GET /health",
            "config": "GET /config",
        },
    }


@app.get("/health")
async def health():
    ingestion, _ = get_pipelines()
    return {"status": "ok", "indexed_chunks": ingestion.vector_store.count()}


@app.get("/config")
async def show_config():
    ingestion, _ = get_pipelines()
    cfg = ingestion.config
    return {
        "embedding_provider": cfg.embedding.provider,
        "vector_store": cfg.vector_store.backend,
        "llm_provider": cfg.llm.provider,
        "swag_profile_mode": cfg.swag.swag_profile_mode,
        "top_k": cfg.top_k,
        "retrieval_candidates": cfg.retrieval_candidates,
        "scoring": cfg.scoring.model_dump(),
    }


@app.post("/ingest", response_model=list[IngestResult])
async def ingest_documents(request: IngestRequest):
    ingestion, _ = get_pipelines()
    directory = Path(request.directory) if request.directory else ingestion.config.data_dir

    if not directory.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")

    results = ingestion.ingest_directory(directory, generate_swag=request.generate_swag)
    ingestion.save_index()
    return results


@app.post("/query", response_model=AnswerResult)
async def query_documents(request: QueryRequest):
    _, retrieval = get_pipelines()

    if retrieval.vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. POST /ingest first.",
        )

    result = retrieval.query(
        question=request.question,
        top_k=request.top_k,
        generate_answer=request.generate_answer,
        use_swag=request.use_swag,
    )
    return result.model_copy(
        update={
            "baseline_results": _trim_results(
                result.baseline_results,
                include_text=request.include_text,
                include_explanation=request.include_explanation,
            ),
            "swag_results": _trim_results(
                result.swag_results,
                include_text=request.include_text,
                include_explanation=request.include_explanation,
            ),
        }
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_retrieval(request: QueryRequest):
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
        baseline=_trim_results(
            result.baseline_results,
            include_text=request.include_text,
            include_explanation=request.include_explanation,
        ),
        swag=_trim_results(
            result.swag_results,
            include_text=request.include_text,
            include_explanation=request.include_explanation,
        ),
        swag_meta=result.swag_meta,
    )
