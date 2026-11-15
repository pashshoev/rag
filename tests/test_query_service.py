from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.api.query.service import retrieve
from rag.shared.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeQdrant:
    def __init__(self, *, has_collection: bool = True) -> None:
        self.has_collection = has_collection
        self.last_query = None

    def get_collections(self):  # type: ignore[no-untyped-def]
        names = ["wikipedia"] if self.has_collection else []
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in names]
        )

    def query_points(self, collection_name: str, query, limit: int, with_payload: bool):  # noqa: ANN001
        self.last_query = {
            "collection_name": collection_name,
            "query": query,
            "limit": limit,
            "with_payload": with_payload,
        }
        point = SimpleNamespace(
            score=0.91,
            payload={
                "text": "RAG retrieves documents before generating.",
                "chunk_id": "retrieval-augmented_generation::0",
                "doc_id": "retrieval-augmented_generation",
                "chunk_index": 0,
                "title": "Retrieval-augmented generation",
                "source_url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                "source_path": "/app/data/corpus/retrieval-augmented_generation.txt",
                "filename": "retrieval-augmented_generation.txt",
            },
        )
        return SimpleNamespace(points=[point])


def test_retrieve_returns_hits() -> None:
    client = FakeQdrant()
    result = retrieve(
        client=client,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        collection_name="wikipedia",
        query="What is RAG?",
        top_k=3,
    )
    assert result.collection == "wikipedia"
    assert result.top_k == 3
    assert len(result.hits) == 1
    assert result.hits[0].score == 0.91
    assert "RAG" in result.hits[0].text
    assert result.hits[0].doc_id == "retrieval-augmented_generation"
    assert client.last_query["limit"] == 3
    assert result.timings.embedding_ms >= 0
    assert result.timings.retrieval_ms >= 0
    assert result.timings.generation_ms == 0
    assert isinstance(result.timings.end2end_ms, int)
    assert result.timings.end2end_ms >= result.timings.embedding_ms


def test_retrieve_requires_collection() -> None:
    with pytest.raises(FileNotFoundError, match="indexer"):
        retrieve(
            client=FakeQdrant(has_collection=False),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),
            collection_name="wikipedia",
            query="What is RAG?",
        )


def test_retrieve_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        retrieve(
            client=FakeQdrant(),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),
            collection_name="wikipedia",
            query="   ",
        )
