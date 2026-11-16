# SWAG — Semantic Weighted Augmented Generation

A Python framework that embeds a **SWAG retrieval enhancement layer** into a traditional RAG pipeline. SWAG improves retrieval ranking by scoring whether a user query aligns with the **contextual semantic weight profile** of each chunk — without modifying the original document text.

**v0.3.3** — always-on multi-candidate filtering, README chrome classified as nav, nav/title noise gate, noise ratio default **0.62**. Installable CLI. Deploy **locally, on-prem, or cloud** — see [docs/DEPLOY.md](docs/DEPLOY.md).

**Docs:** [Build & Usage](docs/BUILD_AND_USAGE.md) · [Architecture diagram](docs/architecture-diagram.html) · [Deploy modes](docs/DEPLOY.md)

## What is SWAG?

SWAG generates structured semantic metadata for each document chunk:

```json
{
  "detected_domain": "finance",
  "primary_concepts": [],
  "domain_terms": [],
  "direct_synonyms": [],
  "adjacent_concepts": [],
  "business_goals": [],
  "user_intents": [],
  "negative_contexts": [],
  "confidence": 0.0,
  "weighted_terms": []
}
```

**Critical design principle:** SWAG metadata is stored **separately** from chunk text. Synonyms and related terms are **weighted contextual hints for reranking only** — never appended to chunks and never treated as source facts.

## How SWAG Differs from Traditional RAG

| Traditional RAG | SWAG-Enhanced RAG |
|-----------------|-------------------|
| Embed chunk text → vector search → return top-K | Same baseline retrieval |
| Ranking = vector similarity only | Ranking = vector + semantic weight profile alignment |
| No disambiguation guardrails | Domain detection + negative contexts prevent false matches |
| Synonyms sometimes injected into chunks | Synonyms are metadata only, never source content |

## How SWAG Differs from CRAG

**CRAG (Corrective RAG)** evaluates retrieval quality and may trigger web search or document re-retrieval when confidence is low. It focuses on *correcting* bad retrieval after the fact.

**SWAG** focuses on *enhancing* retrieval ranking upfront by comparing query and chunk semantic profiles with weighted term matching. SWAG does not replace chunk content or perform corrective web searches — it reranks candidates using semantic alignment scores combined with vector similarity.

```
CRAG:  Retrieve → Evaluate confidence → Correct if low → Generate
SWAG:  Retrieve → SWAG rerank → Generate (original chunks unchanged)
```

In combined deployments (see [langrag](https://github.com/emguru/langrag)), SWAG can sit **before** a corrective-RAG confidence step — better initial ranking means fewer junk chunks graded and fewer false passes from page chrome.

## Relevance proof — what compare mode shows (honest)

This is not a vanity metric. It is how we validate that retrieval is **useful**, not just **similar**:

| Mode | What it proves |
|------|----------------|
| **Vector only** | Classic RAG — embedding similarity. README chrome, nav, and wrong-domain neighbors often rank high. |
| **SWAG** (`swag query --compare`) | Rerank + noise filter — body promoted, duplicates dropped, one chunk per document by default. |
| **CRAG alone** | Grades whatever vector returned — junk already in top-K can still pass as ambiguous. |
| **SWAG → CRAG** | Clean candidates first (≤5), then grade. Empty results mean the corpus could not support a confident answer. |

Portable SWAG proves the **rerank layer** via qrel suites (`swag validate --seed`).

**Try it:** `swag query "how does rag work" --compare` on lesson HTML.

## Architecture

See the interactive diagram: **[docs/architecture-diagram.html](docs/architecture-diagram.html)**

```
Ingest:  Documents → Chunk → SWAG profile (metadata) → Embed original text → Index
Query:   User query → Embed → Vector search (candidates) → SWAG rerank → Top-K → Answer
```

**Rerank pipeline (v0.3):** gate check → ensure profiles → dedupe → score alignment → adaptive blend → noise gate → per-source diversity cap.

## Quick start

Requires **Python 3.11+**.

```bash
git clone <repo-url> SWAG
cd SWAG

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
cp .env.example .env

swag ingest                 # index ./data (recursive)
swag query "How does three-way matching work?" --compare
swag validate --seed --recreate --suite all
swag serve --port 8080
```

Without an API key, the framework uses local `sentence-transformers` embeddings, heuristic SWAG profiles, and `MockLLMClient` — sufficient for demo and testing.

**Docker:**

```bash
docker compose up --build
# http://localhost:8080
```

## Product CLI (`swag`)

| Command | Purpose |
|---------|---------|
| `swag ingest [path]` | Index files or directories (FAISS default) |
| `swag query "…"` | Retrieve + optional answer |
| `swag query "…" --compare` | Baseline vs SWAG side-by-side |
| `swag validate --seed` | Run portable qrel proof suites |
| `swag serve` | Start FastAPI on `:8080` |

**v0.3.3 highlights:** always-on filter on multi-candidate pools; README chrome → nav; nav/title hard gate; noise ratio 0.62. **v0.3.2:** relevance-compare narrative aligned with kube showcase. **v0.3:** structural markdown split (nav vs body), confidence-gated rerank, strict noise filter (1 chunk/doc, no backfill padding), `pip install -e .`, `swag` CLI.

## Project structure

```
swag/
├── app.py              # FastAPI demo API (:8080)
├── config.py           # Configuration (env vars, scoring, SWAG tuning)
├── settings.py         # Kube-compatible settings bridge
├── ingest.py           # Document ingestion pipeline
├── cli.py              # Product CLI (swag ingest|query|serve|validate)
├── gating.py           # Always-on multi-candidate filter (+ optional gate skip)
├── structure.py        # Markdown nav/body segmentation
├── chunk_quality.py    # Chunk-kind detection + multipliers
├── embeddings.py       # sentence-transformers / OpenAI embedders
├── vector_store.py     # FAISS (default) + Qdrant
├── profile.py          # SWAG profile generation (LLM / heuristic)
├── heuristic.py        # Zero-cost heuristic profiles
├── profiles.py         # Ensure profiles on search candidates
├── scoring.py          # Query-chunk profile alignment scoring
├── rerank.py           # SWAG rerank orchestrator
├── postprocess.py      # Dedup, noise gate, diversity, adaptive blend
├── adapters.py         # DocumentChunk ↔ dict hit conversion
├── validation.py       # Profile sanitization + grounding
├── retrieval.py        # Baseline + SWAG reranking pipeline
├── llm_client.py       # OpenAI / Ollama / Mock LLM
├── schemas.py          # Pydantic models
└── utils.py            # Helpers
docs/
├── BUILD_AND_USAGE.md  # Full build & usage guide
├── architecture-diagram.html
└── DEPLOY.md           # Deployment mode quick reference
data/                   # Place documents here
indexes/                # FAISS index + SWAG metadata sidecar
examples/demo.py        # CLI demo workflow
tests/                  # Offline pytest suite
```

## Usage

Full examples: [docs/BUILD_AND_USAGE.md](docs/BUILD_AND_USAGE.md)

### CLI demo

Shows scores and source files only. Use `--verbose` for ranking deltas.

```bash
python examples/demo.py
python examples/demo.py --query "How does vendor invoice matching work?"
python examples/demo.py --reindex
python examples/demo.py --verbose   # ranking deltas + swag_meta
```

### Python API

```python
from swag import IngestionPipeline, RetrievalPipeline, SWAGConfig

config = SWAGConfig.from_env()
config.ensure_dirs()
pipeline = IngestionPipeline(config=config)
pipeline.ingest_directory()
pipeline.save_index()

retrieval = RetrievalPipeline(
    config=config,
    embedder=pipeline.embedder,
    vector_store=pipeline.vector_store,
    profile_generator=pipeline.profile_generator,
)
result = retrieval.query("How does three-way matching work?")
```

### FastAPI

Responses omit chunk text and explanations by default. Pass `include_text` / `include_explanation` to opt in.

```bash
curl -X POST http://localhost:8080/compare \
  -H 'Content-Type: application/json' \
  -d '{"question": "How does three-way matching work?", "top_k": 5}'
```

## Scoring formula

```
final_score = (vector_similarity × VECTOR_WEIGHT)
            + (swag_score × SWAG_WEIGHT)
            + (metadata_boost × METADATA_WEIGHT)
```

Then scaled by chunk-kind quality multiplier (title / nav / boilerplate down-ranked).

**Defaults** (configurable via `.env`):

| Weight | Default | Purpose |
|--------|---------|---------|
| `VECTOR_WEIGHT` | 0.65 | Embedding cosine similarity |
| `SWAG_WEIGHT` | 0.30 | Semantic profile alignment |
| `METADATA_WEIGHT` | 0.05 | Domain match, filename relevance |

When `SWAG_ADAPTIVE_BLEND=true`, weights adjust per query based on vector score distribution.

## Guardrails

1. **No isolated keyword expansion** — ambiguous words like "Apple" require full chunk context
2. **Concept expansion, not keyword stuffing** — multi-word phrases preferred
3. **Domain detection first** — profiles include `detected_domain`
4. **Negative contexts** — e.g. "Apple silicon" includes negatives like "fruit", "orchard"
5. **Synonyms are not facts** — stored as weighted metadata with explicit retrieval-only purpose
6. **Grounding checks** — ungrounded generated terms receive reduced weight

## Backends

| Component | Default | Alternatives |
|-----------|---------|--------------|
| Embeddings | `sentence-transformers` | `openai` |
| Vector store | `faiss` (local disk) | `qdrant` |
| SWAG profiles | `heuristic` ($0) | `llm` (OpenAI / Ollama) |
| LLM answers | Mock (no key) | `openai`, `ollama` |

Extend via `BaseVectorStore` and `BaseLLMClient` factories — see [docs/BUILD_AND_USAGE.md](docs/BUILD_AND_USAGE.md#extending-backends).

## Environment variables

See `.env.example` for all options. Key settings:

```env
EMBEDDING_PROVIDER=sentence-transformers
VECTOR_STORE_BACKEND=faiss
SWAG_PROFILE_MODE=heuristic
VECTOR_WEIGHT=0.65
SWAG_WEIGHT=0.30
METADATA_WEIGHT=0.05
RETRIEVAL_CANDIDATES=20
TOP_K=10
```

## Tests

```bash
pytest tests/ -q
```

## License

**Proprietary — source-visible, limited grant.** This is not open source.

You may view, evaluate, and use the Software for personal learning, research, and internal non-commercial purposes. Commercial use, redistribution, and production deployment require a [commercial license](COMMERCIAL-LICENSE.md).

See [LICENSE](LICENSE) for full terms. **Pricing:** [PRICING.md](PRICING.md) · [Pricing page](blog/pricing.html)

Inquiries: sales@automatico.llc
