#!/usr/bin/env python3
"""SWAG product CLI — ingest, query, serve, validate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ROOT / "benchmarks"


def _bootstrap() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))


def cmd_ingest(args: argparse.Namespace) -> int:
    from swag.config import SWAGConfig
    from swag.ingest import IngestionPipeline

    config = SWAGConfig.from_env()
    config.ensure_dirs()
    pipeline = IngestionPipeline(config=config)

    target = Path(args.path) if args.path else config.data_dir
    if args.recreate and config.vector_store.backend == "faiss":
        for path in (
            config.vector_store.index_path,
            config.vector_store.index_path.with_suffix(".meta.pkl"),
            config.vector_store.index_path.with_suffix(".swag.json"),
        ):
            if path.exists():
                path.unlink()

    if target.is_file():
        results = [pipeline.ingest_file(target, generate_swag=not args.no_swag)]
    else:
        results = pipeline.ingest_directory(target, generate_swag=not args.no_swag)

    pipeline.save_index()
    total = sum(r.num_chunks for r in results)
    print(json.dumps({"files": len(results), "chunks": total, "index": str(config.vector_store.index_path)}, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from swag.config import SWAGConfig
    from swag.ingest import IngestionPipeline
    from swag.retrieval import RetrievalPipeline

    config = SWAGConfig.from_env()
    ingestion = IngestionPipeline(config=config)
    if not ingestion.load_index():
        print("No index found. Run: swag ingest", file=sys.stderr)
        return 2

    retrieval = RetrievalPipeline(
        config=config,
        embedder=ingestion.embedder,
        vector_store=ingestion.vector_store,
        profile_generator=ingestion.profile_generator,
    )

    if args.compare:
        baseline = retrieval.retrieve_baseline(args.question, top_k=args.top_k)
        swag_results, swag_meta = retrieval.retrieve_with_swag(args.question, top_k=args.top_k)
        print(f"Query: {args.question}\n")
        print("--- Baseline top-3 ---")
        for row in baseline[:3]:
            print(f"  [{row.final_score:.3f}] {row.source_document[:60]}")
        print("--- SWAG top-3 ---")
        for row in swag_results[:3]:
            print(f"  [{row.final_score:.3f}] {row.source_document[:60]}")
        if swag_meta.get("gate_skipped"):
            print(f"\n(gated skip: {swag_meta.get('gate_reason')})")
        return 0

    result = retrieval.query(
        args.question,
        top_k=args.top_k,
        generate_answer=not args.no_answer,
        use_swag=not args.baseline,
    )
    print(f"Query: {result.query}")
    if result.answer:
        print(f"\nAnswer: {result.answer}\n")
    rows = result.swag_results if not args.baseline else result.baseline_results
    for row in rows[: args.top_k]:
        preview = (row.raw_text or row.source_document)[:100]
        print(f"  [{row.final_score:.3f}] {preview}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "swag.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from validation_suite_standalone import load_sheet, run_suite
    from swag.config import SWAGConfig
    from swag.retrieval import RetrievalPipeline
    from swag.ingest import IngestionPipeline

    config = SWAGConfig.from_env()
    if args.index_dir:
        config.vector_store.index_path = Path(args.index_dir) / "faiss_index"

    ingestion = IngestionPipeline(config=config)
    if args.seed:
        from seed_validation_corpus import wipe_faiss_index

        fixtures_root = Path(args.fixtures_root)
        if args.recreate:
            wipe_faiss_index(config)
        manifest_path = Path(args.manifest)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        for entry in manifest.get("fixtures") or []:
            seed_key = str(entry["seed_key"])
            topic = str(entry.get("topic") or "")
            path = fixtures_root / str(entry["path"])
            ingestion.ingest_fixture(path, seed_key=seed_key, topic=topic or None)
        for entry in manifest.get("eval_fixtures") or []:
            seed_key = str(entry["seed_key"])
            topic = str(entry.get("topic") or "")
            path = ROOT / str(entry["path"])
            ingestion.ingest_fixture(path, seed_key=seed_key, topic=topic or None)
        ingestion.save_index()

    if not ingestion.load_index():
        print("Index missing after seed.", file=sys.stderr)
        return 2

    pipeline = RetrievalPipeline(
        config=config,
        embedder=ingestion.embedder,
        vector_store=ingestion.vector_store,
        profile_generator=ingestion.profile_generator,
    )

    sheets: list[tuple[str, Path]] = []
    if args.suite in ("rerank", "all"):
        sheets.append(("rerank", Path(args.rerank)))
    if args.suite in ("holdout", "all"):
        sheets.append(("holdout_fixtures", Path(args.holdout)))
    if args.suite in ("hard", "all"):
        sheets.append(("hard_rerank", Path(args.hard)))

    exit_code = 0
    for name, path in sheets:
        if not path.is_file():
            continue
        report = run_suite(pipeline, load_sheet(path))
        print(report["summary"]["report_text"])
        print()
        if report["summary"]["verdict"] != "PASS":
            exit_code = 1
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swag", description="SWAG RAG product CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest documents into local index")
    ingest.add_argument("path", nargs="?", help="File or directory (default: DATA_DIR)")
    ingest.add_argument("--recreate", action="store_true", help="Wipe FAISS index first")
    ingest.add_argument("--no-swag", action="store_true", help="Skip SWAG profile generation")
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Query indexed corpus")
    query.add_argument("question", help="Natural language question")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--compare", action="store_true", help="Show baseline vs SWAG")
    query.add_argument("--baseline", action="store_true", help="Baseline only")
    query.add_argument("--no-answer", action="store_true", help="Retrieval only")
    query.set_defaults(func=cmd_query)

    serve = sub.add_parser("serve", help="Run FastAPI server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    validate = sub.add_parser("validate", help="Run qrel validation suites")
    validate.add_argument("--suite", choices=("rerank", "holdout", "hard", "all"), default="all")
    validate.add_argument("--seed", action="store_true", help="Seed fixtures before running")
    validate.add_argument("--recreate", action="store_true")
    validate.add_argument("--fixtures-root", default=str(ROOT.parent / "kube_rag_langg" / "fixtures" / "corpus"))
    validate.add_argument("--manifest", default=str(BENCHMARKS / "validation_manifest.json"))
    validate.add_argument("--index-dir", default=str(ROOT / "indexes" / "validation"))
    validate.add_argument("--rerank", default=str(BENCHMARKS / "validation_qrels_rerank.json"))
    validate.add_argument("--holdout", default=str(BENCHMARKS / "holdout_fixtures_qrels.json"))
    validate.add_argument("--hard", default=str(BENCHMARKS / "hard_rerank_qrels.json"))
    validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
