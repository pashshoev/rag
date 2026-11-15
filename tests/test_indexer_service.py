from __future__ import annotations

from pathlib import Path

import pytest

from rag.api.indexer.service import _point_id, index_corpus
from rag.shared.embeddings.base import Embedder


class FakeEmbedder(Embedder):
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[float(i)] * self._dimension for i, _ in enumerate(texts)]


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, list] = {}
        self.deleted: list[str] = []

    def get_collections(self):  # type: ignore[no-untyped-def]
        class _Cols:
            def __init__(self, names: list[str]) -> None:
                self.collections = [type("C", (), {"name": n})() for n in names]

        return _Cols(list(self.collections))

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.collections.pop(name, None)

    def create_collection(self, collection_name: str, vectors_config) -> None:  # noqa: ANN001
        self.collections[collection_name] = []

    def upsert(self, collection_name: str, points) -> None:  # noqa: ANN001
        self.collections.setdefault(collection_name, []).extend(points)


def test_point_id_is_stable() -> None:
    assert _point_id("doc::0") == _point_id("doc::0")
    assert _point_id("doc::0") != _point_id("doc::1")


def test_index_corpus_recreate(tmp_path: Path) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text(
        "Title: Sample\nSource: https://example.com\nLicense: test\n---\n\n"
        + ("paragraph about retrieval. " * 40)
        + "\n\n"
        + ("another section with more words. " * 40),
        encoding="utf-8",
    )
    client = FakeQdrant()
    # Pretend collection already exists so recreate path deletes it.
    client.collections["wikipedia"] = []

    result = index_corpus(
        client=client,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        corpus_dir=tmp_path,
        collection_name="wikipedia",
        chunk_size=120,
        chunk_overlap=20,
        recreate=True,
    )

    assert result.docs == 1
    assert result.chunks >= 2
    assert result.points == result.chunks
    assert "wikipedia" in client.deleted
    assert len(client.collections["wikipedia"]) == result.points
    payload = client.collections["wikipedia"][0].payload
    assert "text" in payload
    assert payload["doc_id"] == "sample"
