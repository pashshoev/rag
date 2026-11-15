from __future__ import annotations

import pytest

from rag.shared.embeddings import (
    DEFAULT_MODEL_ID,
    EmbeddingBackend,
    create_embedder,
)


def test_create_embedder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="embedding backend"):
        create_embedder(backend="magic")


def test_ollama_and_triton_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Ollama"):
        create_embedder(backend=EmbeddingBackend.OLLAMA)
    with pytest.raises(NotImplementedError, match="Triton"):
        create_embedder(backend=EmbeddingBackend.TRITON, base_url="http://localhost:8000")


@pytest.mark.slow
def test_in_process_embed_smoke() -> None:
    embedder = create_embedder(
        backend=EmbeddingBackend.IN_PROCESS,
        model_id=DEFAULT_MODEL_ID,
    )
    vectors = embedder.embed(["chunk about vector search"])
    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimension
    assert embedder.dimension == 384
