"""Swappable embedding backends (in-process MVP; Ollama/Triton later)."""

from rag.shared.embeddings.base import (
    DEFAULT_MODEL_ID,
    Embedder,
    EmbeddingBackend,
    create_embedder,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "Embedder",
    "EmbeddingBackend",
    "create_embedder",
]
