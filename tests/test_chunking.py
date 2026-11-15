from __future__ import annotations

from pathlib import Path

import pytest

from rag.shared import CharacterChunker, Document, TextFolderSource

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "data" / "corpus"


def test_character_chunker_respects_size_and_metadata() -> None:
    doc = Document(
        id="demo",
        text=("alpha " * 50) + "\n\n" + ("beta " * 50),
        metadata={"source_path": "demo.txt"},
    )
    chunker = CharacterChunker(chunk_size=80, overlap=20)
    chunks = chunker.split(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 80 for c in chunks)
    assert chunks[0].id == "demo::0"
    assert chunks[0].metadata["doc_id"] == "demo"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["size_unit"] == "characters"
    assert chunks[0].metadata["source_path"] == "demo.txt"


def test_character_chunker_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=100, overlap=100)


def test_text_folder_source_and_corpus_smoke() -> None:
    if not CORPUS.exists():
        pytest.skip("corpus not present — run scripts/download_wikipedia.py")

    source = TextFolderSource(CORPUS)
    docs = list(source.iter_documents())
    assert len(docs) >= 1

    rag_doc = next(d for d in docs if d.id == "retrieval-augmented_generation")
    assert "Retrieval-augmented generation" in (rag_doc.metadata.get("title") or "")
    assert not rag_doc.text.startswith("Title:")
    assert "RAG" in rag_doc.text or "retrieval" in rag_doc.text.lower()

    chunker = CharacterChunker(chunk_size=800, overlap=100)
    chunks = chunker.split(rag_doc)
    assert len(chunks) > 1
    assert chunks[0].metadata["source_url"].startswith("https://")
    # Rough upper bound: step ≈ chunk_size - overlap; allow slack for boundaries.
    max_expected = len(rag_doc.text) // (chunker.chunk_size - chunker.overlap) + 5
    assert len(chunks) <= max_expected
