# SWAG deployment modes

SWAG runs anywhere Python 3.11+ runs: laptop, on-prem server, or cloud VM/Kubernetes.

**Full guide:** [BUILD_AND_USAGE.md](./BUILD_AND_USAGE.md) · **Diagrams:** [architecture-diagram.html](./architecture-diagram.html)

---

## 1. Local development (zero API cost)

```bash
cd SWAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python examples/demo.py
uvicorn swag.app:app --reload --port 8080
```

| Setting | Value |
|---------|--------|
| Embeddings | `sentence-transformers` (local CPU) |
| Vector store | FAISS on disk (`indexes/`) |
| SWAG profiles | `SWAG_PROFILE_MODE=heuristic` |
| Answers | Mock LLM unless `OPENAI_API_KEY` is set |

---

## 2. Docker (local or single VM)

```bash
cp .env.example .env
docker compose up --build
# API: http://localhost:8080
```

Volumes persist `data/` and `indexes/` on the host.

```bash
curl -X POST http://localhost:8080/ingest -H 'Content-Type: application/json' -d '{}'
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is SWAG?", "top_k": 5}'
```

---

## 3. On-prem with Ollama

```bash
# .env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://your-ollama-host:11434
SWAG_PROFILE_MODE=llm          # optional: LLM-generated profiles
EMBEDDING_PROVIDER=sentence-transformers
VECTOR_STORE_BACKEND=faiss
```

Run the API on your network; data stays on your infrastructure. Ollama traffic stays on your LAN.

For Docker reaching a host Ollama:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

---

## 4. Cloud / production-style (Qdrant + OpenAI)

```bash
cp .env.example .env
# Set OPENAI_API_KEY; optionally SWAG_PROFILE_MODE=llm

docker compose --profile qdrant up --build
```

| Setting | Value |
|---------|--------|
| `VECTOR_STORE_BACKEND` | `qdrant` |
| `QDRANT_URL` | `http://qdrant:6333` (compose) or managed Qdrant URL |
| `EMBEDDING_PROVIDER` | `openai` |
| `LLM_PROVIDER` | `openai` |

Deploy the `swag-qdrant` container to ECS, EKS, Fly.io, Railway, etc. Point `QDRANT_URL` at Qdrant Cloud or a self-hosted cluster.

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service info + indexed chunk count |
| GET | `/health` | Liveness |
| GET | `/config` | Active backends and SWAG mode |
| POST | `/ingest` | Index `data/` directory |
| POST | `/query` | RAG query (`use_swag`, `include_text`, `include_explanation`) |
| POST | `/compare` | Baseline vs SWAG side-by-side |

`POST /query` and `POST /compare` omit chunk text and explanations unless `include_text` / `include_explanation` are `true`.

```bash
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is SWAG?", "top_k": 5}'
```

---

## Library usage (embed in your app)

```python
from swag import IngestionPipeline, RetrievalPipeline, SWAGConfig

cfg = SWAGConfig.from_env()
cfg.ensure_dirs()

ingest = IngestionPipeline(config=cfg)
ingest.ingest_directory(cfg.data_dir)
ingest.save_index()

retrieval = RetrievalPipeline(
    config=cfg,
    embedder=ingest.embedder,
    vector_store=ingest.vector_store,
    profile_generator=ingest.profile_generator,
)
result = retrieval.query("What is SWAG?", use_swag=True)
print(f"{result.swag_results[0].source_document}  final={result.swag_results[0].final_score:.3f}")
```

---

## Related: langrag

This repo is the **portable SWAG core**. The [langrag](https://github.com/emguru/langrag) reference stack adds AWS ingest, hybrid search, and an eval harness around the same retrieval ideas.
