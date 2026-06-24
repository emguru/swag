#!/usr/bin/env bash
# Run portable SWAG rerank validation (FAISS, no kube HTTP).
set -euo pipefail

SWAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${SUITE:-all}"
INDEX_DIR="${INDEX_DIR:-${SWAG_ROOT}/indexes/validation}"

cd "${SWAG_ROOT}"
PYTHONPATH="${SWAG_ROOT}:${SWAG_ROOT}/benchmarks" \
  SWAG_PROFILE_MODE=heuristic \
  CHUNK_SIZE=1000 \
  CHUNK_OVERLAP=100 \
  INDEXES_DIR="${SWAG_ROOT}/indexes/validation" \
  "${SWAG_ROOT}/.venv/bin/python" "${SWAG_ROOT}/benchmarks/validation_suite_standalone.py" \
  --suite "${SUITE}" \
  --index-dir "${INDEX_DIR}"
