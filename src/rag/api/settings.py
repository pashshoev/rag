"""Process settings from env — wiring only; passed into factories explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag.shared.embeddings import DEFAULT_MODEL_ID, EmbeddingBackend
from rag.shared.generation import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus"
DEFAULT_COLLECTION = "wikipedia"


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = DEFAULT_COLLECTION
    corpus_dir: Path = DEFAULT_CORPUS
    embedding_backend: str = EmbeddingBackend.IN_PROCESS.value
    embedding_model: str = DEFAULT_MODEL_ID
    embedding_base_url: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 100
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str = "ollama"


def load_settings() -> Settings:
    base_url = (os.getenv("EMBEDDING_BASE_URL") or "").strip() or None
    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        collection_name=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION).strip(),
        corpus_dir=Path(os.getenv("CORPUS_DIR", str(DEFAULT_CORPUS))).expanduser(),
        embedding_backend=os.getenv(
            "EMBEDDING_BACKEND", EmbeddingBackend.IN_PROCESS.value
        ).strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_ID).strip(),
        embedding_base_url=base_url,
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip(),
        llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "ollama").strip() or "ollama",
    )
