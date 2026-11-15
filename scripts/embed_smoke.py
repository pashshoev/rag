#!/usr/bin/env python3
"""Smoke-test the Embedder (env wires args; factory only constructs)."""

from __future__ import annotations

import os

from rag.shared.embeddings import DEFAULT_MODEL_ID, create_embedder


def main() -> None:
    backend = os.getenv("EMBEDDING_BACKEND", "in_process")
    model_id = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_ID)
    base_url = os.getenv("EMBEDDING_BASE_URL") or None
    print(f"backend={backend} model={model_id}")
    embedder = create_embedder(backend=backend, model_id=model_id, base_url=base_url)
    vectors = embedder.embed(["retrieval augmented generation", "hello world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == embedder.dimension
    assert len(vectors[1]) == embedder.dimension
    print(f"ok — dimension={embedder.dimension}, first_dims={vectors[0][:4]}")


if __name__ == "__main__":
    main()
