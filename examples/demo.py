#!/usr/bin/env python3
"""
SWAG demo workflow — ingest, index, query, compare baseline vs SWAG reranking.

Usage:
    python examples/demo.py
    python examples/demo.py --query "How does vendor invoice matching work?"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from swag.config import SWAGConfig
from swag.ingest import IngestionPipeline
from swag.retrieval import RetrievalPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def print_results(title: str, results: list) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)
    for r in results:
        page = f" p.{r.page_number}" if r.page_number else ""
        print(
            f"  {r.rank}. {r.source_document}{page}"
            f"  |  vec {r.vector_similarity:.3f}"
            f"  swag {r.swag_score:.3f}"
            f"  final {r.final_score:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="SWAG RAG demo")
    parser.add_argument(
        "--query",
        default="How does three-way matching work in accounts payable?",
        help="Question to ask",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--reindex", action="store_true", help="Force re-ingestion")
    parser.add_argument("--no-answer", action="store_true", help="Skip answer generation")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show ranking deltas and SWAG metadata (default: scores + answer only)",
    )
    args = parser.parse_args()

    config = SWAGConfig.from_env()
    config.ensure_dirs()

    print("\n🚀 SWAG Demo — Semantic Weighted Augmented Generation")
    print(f"   Data dir: {config.data_dir}")
    print(f"   Scoring: vector={config.scoring.vector_weight}, "
          f"swag={config.scoring.swag_weight}, meta={config.scoring.metadata_weight}")

    # Initialize pipelines
    ingestion = IngestionPipeline(config=config)
    retrieval = RetrievalPipeline(
        config=config,
        embedder=ingestion.embedder,
        vector_store=ingestion.vector_store,
        profile_generator=ingestion.profile_generator,
    )

    # Load or build index
    loaded = ingestion.load_index() if not args.reindex else False
    if loaded:
        print(f"\n📂 Loaded existing index ({ingestion.vector_store.count()} chunks)")
    else:
        print("\n📥 Ingesting documents from data/...")
        results = ingestion.ingest_directory(generate_swag=True)
        for r in results:
            print(f"   ✓ {r.source_filename}: {r.num_chunks} chunks")
        ingestion.save_index()
        print(f"\n💾 Index saved ({ingestion.vector_store.count()} total chunks)")

    # Query
    print(f"\n❓ Query: {args.query}")
    answer_result = retrieval.query(
        question=args.query,
        top_k=args.top_k,
        generate_answer=not args.no_answer,
    )

    print_results("BASELINE RAG (vector similarity only)", answer_result.baseline_results)
    print_results("SWAG RERANKED (vector + semantic weights)", answer_result.swag_results)

    if args.verbose:
        baseline_ids = [r.chunk_id for r in answer_result.baseline_results]
        swag_ids = [r.chunk_id for r in answer_result.swag_results]
        if baseline_ids != swag_ids:
            print("\n📊 Ranking changed after SWAG reranking")
            for i, (b, s) in enumerate(
                zip(answer_result.baseline_results, answer_result.swag_results), start=1
            ):
                if b.chunk_id != s.chunk_id:
                    print(f"   #{i}: {b.source_document} → {s.source_document}")
        else:
            print("\n📊 Rankings identical (SWAG confirmed vector ordering)")
        if answer_result.swag_meta:
            print(f"   SWAG meta: {answer_result.swag_meta}")

    if answer_result.answer:
        print(f"\n💬 Answer:\n{answer_result.answer}")

    print("\n✅ Demo complete.\n")


if __name__ == "__main__":
    main()
