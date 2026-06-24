#!/usr/bin/env python3
"""Seed validation fixtures into local FAISS index (stable seed_key metadata)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent
ROOT = BENCHMARKS.parent


def wipe_faiss_index(config) -> None:
    index_path = Path(config.vector_store.index_path)
    for path in (
        index_path,
        index_path.with_suffix(".meta.pkl"),
        index_path.with_suffix(".swag.json"),
    ):
        if path.exists():
            path.unlink()
            print(f"removed {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=BENCHMARKS / "validation_manifest.json",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        required=True,
        help="Root directory containing fixture markdown files",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing FAISS index before seeding",
    )
    args = parser.parse_args()

    os.environ.setdefault("SWAG_PROFILE_MODE", "heuristic")
    os.environ.setdefault("CHUNK_SIZE", "1000")
    os.environ.setdefault("CHUNK_OVERLAP", "100")
    os.environ.setdefault("VECTOR_STORE_BACKEND", "faiss")
    os.environ.setdefault("INDEXES_DIR", str(ROOT / "indexes" / "validation"))

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from swag.config import SWAGConfig
    from swag.ingest import IngestionPipeline

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    config = SWAGConfig.from_env()
    config.ensure_dirs()
    if args.recreate:
        wipe_faiss_index(config)

    pipeline = IngestionPipeline(config=config)
    results = []
    for entry in manifest.get("fixtures") or []:
        seed_key = str(entry["seed_key"])
        topic = str(entry.get("topic") or "")
        rel_path = Path(str(entry["path"]))
        path = args.fixtures_root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"fixture missing: {path}")
        out = pipeline.ingest_fixture(
            path,
            seed_key=seed_key,
            topic=topic or None,
            generate_swag=True,
        )
        print(f"  fixture {seed_key}: {out.num_chunks} chunks")
        results.append({"seed_key": seed_key, "chunks": out.num_chunks})

    for entry in manifest.get("eval_fixtures") or []:
        seed_key = str(entry["seed_key"])
        topic = str(entry.get("topic") or "")
        path = ROOT / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"eval fixture missing: {path}")
        out = pipeline.ingest_fixture(
            path,
            seed_key=seed_key,
            topic=topic or None,
            generate_swag=True,
        )
        print(f"  eval {seed_key}: {out.num_chunks} chunks")
        results.append({"seed_key": seed_key, "chunks": out.num_chunks})

    pipeline.vector_store.save()
    summary = {
        "fixtures": len(results),
        "chunks": sum(row["chunks"] for row in results),
        "index_path": str(config.vector_store.index_path),
        "points": pipeline.vector_store.count(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
