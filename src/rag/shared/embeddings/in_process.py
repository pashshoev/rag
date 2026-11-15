"""In-process embeddings via sentence-transformers (MVP default)."""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from rag.shared.embeddings.base import Embedder


class InProcessEmbedder(Embedder):
    """Load a HF embedding model into this Python process."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = SentenceTransformer(model_id)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._dimension = int(dim_fn())

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
