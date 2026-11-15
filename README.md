## Project overview

Retrieval-augmented generation (RAG) service: index text documents into a vector store, retrieve relevant chunks for a query, and generate grounded answers with citations. Built as a FastAPI app with a clear split between ingest, retrieval, and generation so backends (embeddings, LLM, vector DB) can be swapped without rewriting the pipeline.

```text
docs → chunk → embed → Qdrant
                           ↑
user query → embed ────────┘ → top-k context → LLM → answer + citations
```

**Stack & keywords:** Python, FastAPI, Streamlit, Docker Compose, Qdrant, HNSW, sentence-transformers, Hugging Face embeddings, OpenAI-compatible API, Ollama, vLLM-ready, RAG, chunking, vector search, grounded generation, citations

## Prerequisites

- Docker + Docker Compose
- Native [Ollama](docs/ollama.md) on the host (Apple GPU / Metal on Mac)
- (Optional) [uv](https://github.com/astral-sh/uv) + Python 3.11+ for local scripts/tests
- Hugging Face token recommended for model downloads ([settings/tokens](https://huggingface.co/settings/tokens))

## Quick start

Runs **API + Qdrant + Streamlit UI** in Docker. The LLM is **host Ollama** (not in Docker).

```bash
# 1. Host Ollama (once) — see docs/ollama.md
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b          # or: make pull-llm

# 2. App stack
cp .env.example .env             # set HF_TOKEN=... ; optional LLM_MODEL=...
make run                         # docker-compose.yml
```

Then open the UI: [http://localhost:8501](http://localhost:8501) (query + `top_k` → JSON with answer + citations).

| Endpoint | Purpose |
| --- | --- |
| `POST /indexer/run` | Chunk + embed corpus → Qdrant |
| `POST /query/retrieve` | Top-k chunks only (no LLM) |
| `POST /query/ask` | Retrieve + grounded generation + citations |

OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs) · UI: [http://localhost:8501](http://localhost:8501)

```bash
make stop
```

### Full stack (Ollama in Docker)

Portable all-in-Docker demo (API + Qdrant + Ollama + UI). On Mac Docker Desktop, Ollama runs on **CPU** — prefer the default `make run` for GPU.

```bash
make run-full    # docker-compose-full-stack.yml
make stop-full
```

### LLM model tips

Set `LLM_MODEL` in `.env`. For the default stack, pull on the host (`make pull-llm` or `ollama pull …`). Full stack pulls inside Compose on boot.

| Tier | Ollama tag | Notes |
| --- | --- | --- |
| Sub‑1B | `qwen3:0.6b` | Fast smoke tests; weaker grounded RAG |
| ~1–4B | `llama3.2:1b` (default) / `llama3.2:3b` | Better instruction-following for grounded Q&A |
| Sub‑8B | `qwen2.5:7b` | Higher quality if you have headroom |

Same HTTP client works with **vLLM** (or any OpenAI-compatible server): change `LLM_BASE_URL` / `LLM_MODEL` only.

Host Ollama install and lifecycle: [`docs/ollama.md`](docs/ollama.md).

## Repository layout

```text
src/rag/
  api/           # FastAPI app + feature routers
    indexer/     # POST /indexer/*
    query/       # POST /query/retrieve, /query/ask
  shared/        # ingest + model clients (sources, chunking, embeddings, generation)
  ui/            # Streamlit client (HTTP → API)
data/corpus/     # plain-text docs to index
scripts/         # one-off helpers (download corpus, smoke tests)
tests/
docker-compose.yml              # default: API + UI + Qdrant → host Ollama
docker-compose-full-stack.yml   # all-in-Docker including Ollama
Dockerfile.ui
```

Corpus is Wikipedia plain text (CC BY-SA) plus any extra `.txt` you drop in `data/corpus/`. Attribution: [`data/README.md`](data/README.md).

## Extending (high level)

| Goal | Where to look |
| --- | --- |
| New document source (e.g. products) | `DataSource` / `TextFolderSource` in `shared/sources.py` |
| Chunking behavior | `CharacterChunker` in `shared/chunking.py` |
| Swap embedding backend | `Embedder` ABC + `create_embedder()` in `shared/embeddings/` (`EmbeddingBackend`: `in_process`, stubs for `ollama` / `triton`) |
| Swap LLM / provider | `Generator` ABC + `create_generator()` / `OpenAICompatibleGenerator` in `shared/generation.py` — point `LLM_BASE_URL` at Ollama, vLLM, etc. |
| New HTTP features | Add a package under `api/` and `include_router` in `api/app.py` |
| Config wiring | `api/settings.py` + `.env` / Compose `environment` |

Indexer and query stay separate packages so a future split into two services is mostly a deploy change, not a rewrite.

## Local development (optional)

```bash
uv sync --group dev
source .venv/bin/activate
make api-local          # uvicorn with reload; needs reachable Qdrant (+ LLM for /ask)
make ui-local           # Streamlit on :8501 → http://localhost:8000
pytest -q
```

## License / data

Code: use and adapt as you like. Corpus text from English Wikipedia is under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — see [`data/README.md`](data/README.md).
