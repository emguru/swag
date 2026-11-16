# Build and Usage Guide

Deploy, operate, and demo **SWAG** — the portable Semantic Weighted Augmented Generation layer for RAG.

**Diagrams:** [architecture-diagram.html](./architecture-diagram.html) · **Deployment modes:** [DEPLOY.md](./DEPLOY.md)

---

## Table of contents

1. [What this is](#what-this-is)
2. [Prerequisites](#prerequisites)
3. [Quick start (local, $0)](#quick-start-local-0)
4. [Configuration](#configuration)
5. [Ingest documents](#ingest-documents)
6. [Query and compare](#query-and-compare)
7. [SWAG reranking pipeline](#swag-reranking-pipeline)
8. [Deployment modes](#deployment-modes)
9. [Library usage](#library-usage)
10. [Tests and validation](#tests-and-validation)
11. [Extending backends](#extending-backends)
12. [Troubleshooting](#troubleshooting)
13. [Related: langrag](#related-langrag)

---

## What this is

| Piece | Technology |
|-------|------------|
| Core | Python 3.11+ library + FastAPI demo API |
| Embeddings | sentence-transformers (local) or OpenAI |
| Vector store | FAISS (default, disk) or Qdrant (on-prem / cloud) |
| SWAG profiles | `heuristic` (default, $0) or `llm` (OpenAI / Ollama) |
| Reranker | Production-hardened engine (noise gating, profile alignment, diversity cap) |
| Answers | OpenAI, Ollama, or Mock LLM fallback |

**SWAG** is a retrieval enhancement layer — not a replacement for RAG or CRAG. It stores semantic metadata **separately** from chunk text and reranks vector search candidates using weighted profile alignment, noise gating, chunk-kind multipliers, and grounding checks.

**v0.2** ports the kube production reranker: `rerank.py`, `postprocess.py`, `heuristic.py`, `chunk_quality.py`, adaptive blend weights, and per-source diversity caps.

---

## Prerequisites

### Tools

Python 3.11+ · git · (optional) Docker

```bash
python3 --version   # 3.11+
```

### Keys (`.env`)

| Key | Required | Purpose |
|-----|----------|---------|
| *(none)* | — | Heuristic SWAG + local embeddings work offline |
| `OPENAI_API_KEY` | Optional | LLM profiles, OpenAI embeddings, real answers |
| `OLLAMA_BASE_URL` | Optional | On-prem LLM when `LLM_PROVIDER=ollama` |

---

## Quick start (local, $0)

```bash
git clone <repo-url> SWAG
cd SWAG

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

# Ingest sample docs + compare baseline vs SWAG
python examples/demo.py

# Start API on port 8080
uvicorn swag.app:app --reload --port 8080
```

Defaults (no API key):

- **Embeddings:** `sentence-transformers` / `all-MiniLM-L6-v2` (local CPU)
- **Vector store:** FAISS on disk (`indexes/`)
- **SWAG profiles:** `SWAG_PROFILE_MODE=heuristic`
- **Answers:** Mock LLM

Verify:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/compare \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does three-way matching work?", "top_k": 5}'
```

---

## Configuration

Copy `.env.example` → `.env`. Key groups:

### Scoring blend

| Variable | Default | Purpose |
|----------|---------|---------|
| `VECTOR_WEIGHT` | `0.65` | Embedding cosine similarity |
| `SWAG_WEIGHT` | `0.30` | Semantic profile alignment |
| `METADATA_WEIGHT` | `0.05` | Domain match, filename relevance |

Weights must sum to `1.0`.

### SWAG engine

| Variable | Default | Purpose |
|----------|---------|---------|
| `SWAG_ENABLED` | `true` | Master switch for reranking |
| `SWAG_PROFILE_MODE` | `heuristic` | `heuristic` ($0) or `llm` |
| `SWAG_INGEST_PROFILES` | `true` | Generate profiles at ingest |
| `RETRIEVAL_CANDIDATES` | `20` | Vector pool before rerank |
| `TOP_K` | `10` | Final results returned |
| `SWAG_MIN_SCORE_THRESHOLD` | `0.08` | Noise floor |
| `SWAG_NOISE_RELATIVE_VECTOR_RATIO` | `0.65` | Drop weak SWAG vs vector |
| `SWAG_MAX_CHUNKS_PER_SOURCE` | `3` | Per-document diversity cap |
| `SWAG_ADAPTIVE_BLEND` | `true` | Dynamic weight adjustment |

Full tuning knobs: `swag/config.py` → `SwagTuningConfig`.

### Backends

| Variable | Default | Options |
|----------|---------|---------|
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `openai` |
| `VECTOR_STORE_BACKEND` | `faiss` | `qdrant` |
| `LLM_PROVIDER` | `openai` | `ollama`, `mock` |

---

## Ingest documents

Place files in `data/` — supports `.txt`, `.md`, `.pdf`, `.docx`.

### CLI demo

Shows **scores and source files only** by default — no chunk text dumps.

```bash
python examples/demo.py
python examples/demo.py --query "How does vendor invoice matching work?"
python examples/demo.py --reindex       # force rebuild
python examples/demo.py --no-answer     # skip LLM answer
python examples/demo.py --verbose       # ranking deltas + swag_meta
```

Example output:

```
  1. accounts_payable_guide.md  |  vec 0.715  swag 0.374  final 0.504
  2. accounts_payable_guide.md  |  vec 0.732  swag 0.000  final 0.305
```

### Python API

```python
from swag import IngestionPipeline, SWAGConfig

config = SWAGConfig.from_env()
config.ensure_dirs()
pipeline = IngestionPipeline(config=config)

results = pipeline.ingest_directory(generate_swag=True)
pipeline.save_index()

for r in results:
    print(f"{r.source_filename}: {r.num_chunks} chunks")
```

### FastAPI

```bash
curl -X POST http://localhost:8080/ingest \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Optional body: `{"directory": "/path/to/docs", "generate_swag": true}`

**Ingest flow:** load → chunk → SWAG profile (metadata only) → embed original text → index vectors + profiles.

---

## Query and compare

API and demo responses are **quiet by default**: scores, source filenames, and answers — not full chunk text or SWAG match explanations. Opt in when you need detail.

### Response detail flags

| Flag | Where | Default | When `true` |
|------|-------|---------|-------------|
| `--verbose` | CLI demo | off | Ranking deltas + `swag_meta` |
| `include_text` | `POST /query`, `/compare` | `false` | Full `raw_text` per hit |
| `include_explanation` | `POST /query`, `/compare` | `false` | Per-hit SWAG scoring explanations |

The Python library always returns full `RetrievalResult` objects (including `raw_text`) — trimming applies to the FastAPI layer only.

### Python API

```python
from swag import IngestionPipeline, RetrievalPipeline, SWAGConfig

config = SWAGConfig.from_env()
ingestion = IngestionPipeline(config=config)
ingestion.load_index()

retrieval = RetrievalPipeline(
    config=config,
    embedder=ingestion.embedder,
    vector_store=ingestion.vector_store,
    profile_generator=ingestion.profile_generator,
)

result = retrieval.query("How does three-way matching work?")

top = result.swag_results[0]
print(f"{top.source_document}  final={top.final_score:.3f}")
print("Answer:", result.answer)
```

### FastAPI

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service info + indexed chunk count |
| GET | `/health` | Liveness |
| GET | `/config` | Active backends and SWAG mode |
| POST | `/ingest` | Index `data/` (or custom directory) |
| POST | `/query` | RAG query with optional SWAG rerank |
| POST | `/compare` | Baseline vs SWAG side-by-side |

```bash
# SWAG rerank — scores + source only (default)
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does three-way matching work?", "top_k": 5, "use_swag": true}'

# Include full chunk text and SWAG explanations
curl -X POST http://localhost:8080/compare \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does three-way matching work?", "top_k": 5, "include_text": true, "include_explanation": true}'

# Baseline only
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "...", "use_swag": false}'
```

---

## SWAG reranking pipeline

Single index — vectors and `swag_profile` metadata live together (FAISS sidecar or Qdrant payload). Search auto-enriches missing profiles before rerank.

```
Query
  → Embed query
  → Vector search (RETRIEVAL_CANDIDATES)
  → Ensure SWAG profiles (cache / inline generate)
  → Dedupe near-identical chunks
  → Query SWAG profile
  → Score alignment + chunk-kind multipliers + metadata boost
  → Adaptive blend (vector / SWAG / meta weights)
  → Noise gate + per-source diversity cap
  → Top-K results
  → (optional) LLM answer from top chunks
```

### Term category weights

| Category | Multiplier | Role |
|----------|------------|------|
| `primary_concept` | 1.0× | Core topic match |
| `user_intent` | 0.90× | Question intent alignment |
| `domain_term` | 0.85× | Domain vocabulary |
| `business_goal` | 0.75× | Operational goal overlap |
| `direct_synonym` | 0.70× | Retrieval hint only |
| `adjacent_concept` | 0.60× | Related but not primary |
| `negative_context` | penalty | Blocks false-domain matches |

### Guardrails

1. Original chunk text is never modified — profiles are metadata only
2. Synonyms are retrieval hints, not source facts
3. Domain detection before term assignment
4. Grounding checks penalize ungrounded generated terms
5. Chunk-kind multipliers down-rank titles, nav, boilerplate

See [architecture-diagram.html](./architecture-diagram.html) for sequence diagrams.

---

## Deployment modes

| Mode | Command | Backends |
|------|---------|----------|
| **Local dev** | `python examples/demo.py` | FAISS + sentence-transformers + heuristic |
| **Docker** | `docker compose up --build` | Same as local; volumes for `data/` + `indexes/` |
| **On-prem LLM** | Set `LLM_PROVIDER=ollama` | Ollama on LAN; optional `SWAG_PROFILE_MODE=llm` |
| **Cloud** | `docker compose --profile qdrant up --build` | Qdrant + OpenAI embeddings |

Detail: [DEPLOY.md](./DEPLOY.md)

---

## Library usage

Embed SWAG in your own app — import from `swag`:

```python
from swag import (
    SWAGConfig,
    IngestionPipeline,
    RetrievalPipeline,
    rerank_hits_with_swag,
)
```

Use `rerank_hits_with_swag()` directly if you already have vector hits as dicts (kube-compatible shape via `swag/adapters.py`).

---

## Tests and validation

```bash
source .venv/bin/activate
pytest tests/ -q
```

| Test file | Covers |
|-----------|--------|
| `test_swag_heuristic.py` | Heuristic profile generation |
| `test_swag_scoring.py` | Profile alignment scoring |
| `test_swag_postprocess.py` | Dedup, noise gate, diversity |

Offline tests require no API keys.

---

## Extending backends

### Vector stores

Implement `BaseVectorStore` in `swag/vector_store.py`. FAISS and Qdrant are built in; Chroma has a stub.

```env
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
```

### LLM providers

Implement `BaseLLMClient` in `swag/llm_client.py`. Built in: OpenAI, Ollama, Mock.

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No documents indexed` | Run `POST /ingest` or `python examples/demo.py` |
| Rankings unchanged | Query may already be well-served by vectors; try ambiguous domain queries |
| Slow first run | sentence-transformers downloads model on first embed |
| `Unsupported vector store` | Set `VECTOR_STORE_BACKEND=faiss` or `qdrant` |
| Mock answers only | Set `OPENAI_API_KEY` or `LLM_PROVIDER=ollama` |
| Qdrant connection refused | Start Qdrant: `docker compose --profile qdrant up qdrant` |
| Scoring weight error | Ensure `VECTOR_WEIGHT + SWAG_WEIGHT + METADATA_WEIGHT = 1.0` |

---

## Related: langrag

This repo is the portable SWAG core. [langrag](https://github.com/emguru/langrag) is the AWS reference stack (ingest, hybrid search, eval).

---

## Related documentation

| Document | Description |
|----------|-------------|
| [architecture-diagram.html](./architecture-diagram.html) | Interactive Mermaid diagrams |
| [DEPLOY.md](./DEPLOY.md) | Deployment mode quick reference |
| [../README.md](../README.md) | Project overview |
| [../COMMERCIAL-LICENSE.md](../COMMERCIAL-LICENSE.md) | Commercial licensing |
