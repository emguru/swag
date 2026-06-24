#!/usr/bin/env bash
# Seed validation fixtures into local FAISS (standalone SWAG stack).
set -euo pipefail

SWAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUBE_ROOT="${KUBE_ROOT:-$(cd "${SWAG_ROOT}/../kube_rag_langg" 2>/dev/null && pwd || true)}"
FIXTURES_ROOT="${FIXTURES_ROOT:-${KUBE_ROOT}/fixtures/corpus}"
RECREATE="${RECREATE:-1}"

if [[ ! -d "${FIXTURES_ROOT}" ]]; then
  echo "FIXTURES_ROOT missing: ${FIXTURES_ROOT}" >&2
  exit 1
fi

ARGS=(--fixtures-root "${FIXTURES_ROOT}")
if [[ "${RECREATE}" == "1" ]]; then
  ARGS+=(--recreate)
fi

cd "${SWAG_ROOT}"
PYTHONPATH="${SWAG_ROOT}:${SWAG_ROOT}/benchmarks" \
  SWAG_PROFILE_MODE=heuristic \
  CHUNK_SIZE=1000 \
  CHUNK_OVERLAP=100 \
  INDEXES_DIR="${SWAG_ROOT}/indexes/validation" \
  "${SWAG_ROOT}/.venv/bin/python" "${SWAG_ROOT}/benchmarks/seed_validation_corpus.py" "${ARGS[@]}"
