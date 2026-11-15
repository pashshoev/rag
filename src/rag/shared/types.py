"""Core document/chunk types shared across ingest and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single loadable unit of text (file, product row, page, …)."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit derived from a Document."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Timings:
    """RAG pipeline stage durations in milliseconds."""

    end2end_ms: int
    embedding_ms: int
    retrieval_ms: int
    generation_ms: int = 0
