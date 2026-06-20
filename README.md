# SWAG — Semantic Weighted Augmented Generation

A Python framework that embeds a **SWAG retrieval enhancement layer** into a traditional RAG pipeline. SWAG improves retrieval ranking by scoring whether a user query aligns with the **contextual semantic weight profile** of each chunk — without modifying the original document text.

## What is SWAG?

SWAG generates structured semantic metadata for each document chunk:

```json
{
  "primary_concepts": [],
  "domain_terms": [],
  "direct_synonyms": [],
  "adjacent_concepts": [],
  "business_goals": [],
  "user_intents": [],
  "negative_contexts": [],
  "confidence": 0.0
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

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Documents  │────▶│   Chunking   │────▶│  Original Chunks │
│ (.txt/.md/  │     │              │     │  (authoritative) │
│  .pdf/.docx)│     └──────────────┘     └────────┬────────┘
└─────────────┘                                   │
                    ┌──────────────────────────────┼──────────────────┐
                    ▼                              ▼                  ▼
            ┌──────────────┐              ┌──────────────┐   ┌──────────────┐
            │  Embeddings  │              │ SWAG Profile │   │ Vector Store │
            │ (chunk text) │              │  Generation  │   │   (FAISS)    │
            └──────┬───────┘              └──────┬───────┘   └──────┬───────┘
                   │                             │                  │
                   └──────────────┬──────────────┘                  │
                                  ▼                                 │
                         ┌────────────────┐                         │
                         │  Index Storage │◀────────────────────────┘
                         └────────────────┘

Query Flow:
  User Query → Embed → Vector Search (top-K candidates)
                     → Query SWAG Profile
                     → Score alignment with chunk SWAG profiles
                     → final_score = vector×0.65 + swag×0.30 + meta×0.05
                     → Rerank → Generate Answer
```

## Installation

Requires **Python 3.11+**.

```bash
git clone <repo-url> SWAG
cd SWAG

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Optionally set OPENAI_API_KEY for real LLM profiles and answers
```

Without an API key, the framework uses `MockLLMClient` with heuristic SWAG profile extraction — sufficient for local demo and testing.

## Project Structure

```
swag/
├── app.py              # FastAPI demo API
├── config.py           # Configuration (env vars, scoring weights)
├── ingest.py           # Document ingestion pipeline
├── chunking.py         # Configurable chunk size/overlap
├── embeddings.py       # sentence-transformers / OpenAI embedders
├── vector_store.py     # FAISS (default) + stubs for Qdrant/Pinecone/Chroma
├── swag_profile.py     # SWAG metadata generation (LLM)
├── swag_scoring.py     # Query-chunk profile alignment scoring
├── retrieval.py        # Baseline + SWAG reranking pipeline
├── llm_client.py       # Modular LLM (OpenAI default)
├── schemas.py          # Pydantic models
└── utils.py            # Helpers
data/                   # Place documents here
indexes/                # FAISS index persistence
examples/demo.py        # CLI demo workflow
```

## Ingest Documents

Place files in `data/` (supports `.txt`, `.md`, `.pdf`, `.docx`).

### CLI Demo

```bash
python examples/demo.py
python examples/demo.py --query "How does vendor invoice matching work?"
python examples/demo.py --reindex   # force rebuild
```

### Python API

```python
from swag import IngestionPipeline, SWAGConfig

config = SWAGConfig.from_env()
pipeline = IngestionPipeline(config=config)

results = pipeline.ingest_directory()
pipeline.save_index()

for r in results:
    print(f"{r.source_filename}: {r.num_chunks} chunks")
```

### FastAPI

```bash
uvicorn swag.app:app --reload
```

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'
```

## Query

### Python API

```python
from swag import RetrievalPipeline, IngestionPipeline, SWAGConfig

config = SWAGConfig.from_env()
ingestion = IngestionPipeline(config=config)
ingestion.load_index()

retrieval = RetrievalPipeline(
    config=config,
    embedder=ingestion.embedder,
    vector_store=ingestion.vector_store,
)

# Compare baseline vs SWAG
result = retrieval.query("How does three-way matching work?")

print("Baseline:", result.baseline_results[0].explanation)
print("SWAG:", result.swag_results[0].explanation)
print("Answer:", result.answer)
```

### FastAPI

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "How does three-way matching work?", "top_k": 5}'
```

## Scoring Formula

```
final_score = (vector_similarity × VECTOR_WEIGHT)
            + (swag_score × SWAG_WEIGHT)
            + (metadata_boost × METADATA_WEIGHT)
```

**Defaults** (configurable via `.env`):

| Weight | Default | Purpose |
|--------|---------|---------|
| `VECTOR_WEIGHT` | 0.65 | Embedding cosine similarity |
| `SWAG_WEIGHT` | 0.30 | Semantic profile alignment |
| `METADATA_WEIGHT` | 0.05 | Domain match, filename relevance |

SWAG scoring compares query and chunk profiles across weighted term categories:

- `primary_concept` (1.0×)
- `user_intent` (0.90×)
- `domain_term` (0.85×)
- `business_goal` (0.75×)
- `direct_synonym` (0.70×) — retrieval hint only
- `adjacent_concept` (0.60×)
- `negative_context` — applies penalty to prevent false matches

## Guardrails

SWAG implements protections against **semantic pollution**:

1. **No isolated keyword expansion** — ambiguous words like "Apple" require full chunk context
2. **Concept expansion, not keyword stuffing** — multi-word phrases preferred
3. **Domain detection first** — profiles include `detected_domain`
4. **Negative contexts** — e.g., "Apple silicon" includes negatives like "fruit", "orchard"
5. **Synonyms are not facts** — stored as weighted metadata with explicit retrieval-only purpose

## Swap Vector Databases

The vector store is modular via `BaseVectorStore` in `vector_store.py`.

**FAISS (default):**
```env
VECTOR_STORE_BACKEND=faiss
```

**Add Qdrant, Pinecone, or Chroma:**

1. Implement a class extending `BaseVectorStore` (see `FAISSVectorStore` as reference)
2. Register it in `create_vector_store()` factory
3. Set `VECTOR_STORE_BACKEND=qdrant` (or `chroma`, `pinecone`)

Stub classes are provided in `vector_store.py` with implementation guidance.

## Add Another LLM Provider

LLM integration is modular via `BaseLLMClient` in `llm_client.py`.

1. Create a class implementing `complete()`, `complete_json()`, and optionally `generate_answer()`
2. Register in `create_llm_client()` factory
3. Set `LLM_PROVIDER=your_provider` in `.env`

OpenAI is the default. `MockLLMClient` provides offline fallback.

## Environment Variables

See `.env.example` for all options:

```env
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=sentence-transformers
VECTOR_STORE_BACKEND=faiss
VECTOR_WEIGHT=0.65
SWAG_WEIGHT=0.30
METADATA_WEIGHT=0.05
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=10
```

## License

**Proprietary — source-visible, limited grant.** This is not open source.

You may view, evaluate, and use the Software for personal learning, research, and
internal non-commercial purposes. Commercial use, redistribution, and production
deployment require a [commercial license](COMMERCIAL-LICENSE.md).

See [LICENSE](LICENSE) for full terms. **Pricing:** [PRICING.md](PRICING.md) · [Pricing page](blog/pricing.html)

Inquiries: sales@automatico.llc
