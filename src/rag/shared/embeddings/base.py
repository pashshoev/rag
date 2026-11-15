"""Abstract Embedder + factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"


class EmbeddingBackend(str, Enum):
    IN_PROCESS = "in_process"
    OLLAMA = "ollama"
    TRITON = "triton"


class Embedder(ABC):
    """Abstract embedder — backends inherit and implement."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Model name / id used by this backend."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Fixed output vector size."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts; one vector per input (same order)."""


def create_embedder(
    backend: EmbeddingBackend | str = EmbeddingBackend.IN_PROCESS,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str | None = None,
) -> Embedder:
    """Construct an Embedder for the given backend.

    Call sites (app startup / CLI) read env and pass values in; this function
    only builds the object.
    """
    if isinstance(backend, str):
        try:
            backend = EmbeddingBackend(backend.strip().lower())
        except ValueError as exc:
            allowed = ", ".join(b.value for b in EmbeddingBackend)
            raise ValueError(
                f"Invalid embedding backend={backend!r}. Allowed: {allowed}"
            ) from exc

    if backend is EmbeddingBackend.IN_PROCESS:
        from rag.shared.embeddings.in_process import InProcessEmbedder

        return InProcessEmbedder(model_id=model_id)

    if backend is EmbeddingBackend.OLLAMA:
        raise NotImplementedError(
            "Ollama embedding backend is not implemented yet. "
            "Use backend=in_process for the MVP."
            + (f" (base_url={base_url})" if base_url else "")
        )

    if backend is EmbeddingBackend.TRITON:
        raise NotImplementedError(
            "Triton embedding backend is not implemented yet. "
            "Use backend=in_process for the MVP. "
            "Local Mac usually points at a remote Linux+GPU Triton."
            + (f" (base_url={base_url})" if base_url else "")
        )

    raise ValueError(f"Unsupported embedding backend: {backend}")
