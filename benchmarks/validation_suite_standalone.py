#!/usr/bin/env python3
"""Run qrel validation against local FAISS + SWAG rerank (no HTTP, no kube)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BENCHMARKS = Path(__file__).resolve().parent
ROOT = BENCHMARKS.parent


def _strip_internal(hit: dict) -> dict:
    return {key: value for key, value in hit.items() if not key.startswith("_")}


def _compute_lift(baseline_hits: list[dict], swag_hits: list[dict], *, top_k: int) -> dict:
    window = max(1, min(top_k, len(baseline_hits) or top_k))
    improved = 0
    for idx in range(window):
        base_id = baseline_hits[idx].get("id") if idx < len(baseline_hits) else None
        for rank, hit in enumerate(swag_hits[:window], start=1):
            if hit.get("id") == base_id and rank < idx + 1:
                improved += 1
                break
    rank_improved_pct = round(100.0 * improved / window, 1) if window else 0.0
    metrics: list[dict[str, float | str]] = []
    if rank_improved_pct > 0:
        metrics.append({"label": "Higher-ranked results", "value_pct": rank_improved_pct})
    return {
        "pipeline": "swag_rerank",
        "rank_improved_pct": rank_improved_pct,
        "positive_metrics": metrics,
        "has_positive_lift": bool(metrics),
    }


def search_case(
    pipeline,
    case: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    from swag.adapters import chunk_to_hit
    from swag.rerank import rerank_hits_with_swag

    query = str(case["query"])
    topic = case.get("topic")
    candidate_k = max(pipeline.config.retrieval_candidates, top_k)
    candidates_raw = pipeline._vector_search(query, candidate_k, topic=topic)
    pool_hits = [chunk_to_hit(sr.chunk, sr.score) for sr in candidates_raw]
    baseline_raw = [_strip_internal(hit) for hit in pool_hits[:top_k]]

    swag_hits, _meta = rerank_hits_with_swag(
        query,
        pool_hits,
        collection=pipeline.config.vector_store.collection_name,
        top_k=top_k,
        vector_store=pipeline.vector_store,
    )
    hits = [_strip_internal(hit) for hit in swag_hits]
    lift = {"swag_rerank": _compute_lift(baseline_raw, hits, top_k=top_k)}

    return {
        "hits": hits,
        "baseline_hits": baseline_raw,
        "baseline_hits_raw": baseline_raw,
        "lift": lift,
    }


def run_suite(pipeline, sheet: dict[str, Any]) -> dict[str, Any]:
    from validation_eval import evaluate_case, summarize_results

    top_k = int(sheet.get("top_k") or 8)
    cases = list(sheet.get("cases") or [])
    results = [
        evaluate_case(case, search_case(pipeline, case, top_k=top_k), top_k=top_k)
        for case in cases
    ]
    summary = summarize_results(sheet, results)
    return {"sheet_path": sheet.get("_path"), "summary": summary, "cases": results}


def load_sheet(path: Path) -> dict[str, Any]:
    from validation_eval import load_qrels

    sheet = load_qrels(path)
    sheet["_path"] = str(path)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerank",
        type=Path,
        default=BENCHMARKS / "validation_qrels_rerank.json",
    )
    parser.add_argument(
        "--holdout-fixtures",
        type=Path,
        default=BENCHMARKS / "holdout_fixtures_qrels.json",
    )
    parser.add_argument(
        "--suite",
        choices=("rerank", "holdout", "hard", "all"),
        default="all",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        default=BENCHMARKS / "hard_rerank_qrels.json",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "indexes" / "validation",
        help="Directory containing seeded FAISS validation index",
    )
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("SWAG_PROFILE_MODE", "heuristic")
    os.environ.setdefault("CHUNK_SIZE", "1000")
    os.environ.setdefault("CHUNK_OVERLAP", "100")
    os.environ.setdefault("VECTOR_STORE_BACKEND", "faiss")
    os.environ.setdefault("INDEXES_DIR", str(args.index_dir.parent))

    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from swag.config import SWAGConfig
    from swag.retrieval import RetrievalPipeline

    config = SWAGConfig.from_env()
    config.vector_store.index_path = args.index_dir / "faiss_index"
    pipeline = RetrievalPipeline(config=config)
    if not pipeline.vector_store.load():
        print("FAISS index missing — run seed_validation_corpus.py first", file=sys.stderr)
        return 2

    paths: list[tuple[str, Path]] = []
    if args.suite in ("rerank", "all"):
        paths.append(("rerank", args.rerank))
    if args.suite in ("holdout", "all"):
        paths.append(("holdout_fixtures", args.holdout_fixtures))
    if args.suite in ("hard", "all"):
        paths.append(("hard_rerank", args.hard))

    report: dict[str, Any] = {"stack": "swag_standalone_faiss", "suites": {}}
    exit_code = 0
    for name, path in paths:
        sheet = load_sheet(path)
        suite_report = run_suite(pipeline, sheet)
        report["suites"][name] = suite_report
        if not args.json_only:
            print(suite_report["summary"]["report_text"], file=sys.stderr)
            print(file=sys.stderr)
        if suite_report["summary"]["verdict"] != "PASS":
            exit_code = 1

    report["verdict"] = "PASS" if exit_code == 0 else "FAIL"
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
