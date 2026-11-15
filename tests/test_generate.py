from __future__ import annotations

from types import SimpleNamespace

from rag.api.query.generate import SYSTEM_PROMPT, _build_messages, ask
from rag.api.query.service import RetrievedChunk
from rag.shared.embeddings.base import Embedder
from rag.shared.generation import Generator


class FakeEmbedder(Embedder):
    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeGenerator(Generator):
    @property
    def model_id(self) -> str:
        return "fake-llm"

    def complete(self, messages):  # type: ignore[no-untyped-def]
        assert messages[0]["role"] == "system"
        assert "Context:" in messages[1]["content"]
        return "Grounded answer from context."


class FakeQdrant:
    def get_collections(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(collections=[SimpleNamespace(name="wikipedia")])

    def query_points(self, collection_name: str, query, limit: int, with_payload: bool):  # noqa: ANN001
        point = SimpleNamespace(
            score=0.9,
            payload={
                "text": "RAG retrieves documents before generating answers.",
                "chunk_id": "rag::0",
                "doc_id": "rag",
                "chunk_index": 0,
                "title": "Retrieval-augmented generation",
                "source_url": "https://example.com/rag",
                "source_path": "/data/rag.txt",
                "filename": "rag.txt",
            },
        )
        return SimpleNamespace(points=[point])


def test_build_messages_includes_context() -> None:
    hits = [
        RetrievedChunk(
            score=0.9,
            text="hello context",
            chunk_id="a::0",
            doc_id="a",
            chunk_index=0,
            title="Doc A",
            source_url="https://example.com",
            source_path=None,
            filename="a.txt",
        )
    ]
    messages = _build_messages(query="What is it?", hits=hits)
    assert "Doc A" in messages[1]["content"]
    assert "hello context" in messages[1]["content"]
    assert "What is it?" in messages[1]["content"]


def test_ask_returns_answer_and_citations() -> None:
    result = ask(
        client=FakeQdrant(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        generator=FakeGenerator(),
        collection_name="wikipedia",
        query="What is RAG?",
        top_k=3,
    )
    assert result.answer == "Grounded answer from context."
    assert result.model == "fake-llm"
    assert len(result.hits) == 1
    assert result.hits[0].title == "Retrieval-augmented generation"
    assert result.timings.generation_ms >= 0
    assert result.timings.embedding_ms >= 0
    assert result.timings.retrieval_ms >= 0
    assert result.timings.end2end_ms >= result.timings.generation_ms


def test_system_prompt_asks_for_short_answer() -> None:
    assert "short answer" in SYSTEM_PROMPT.lower()
