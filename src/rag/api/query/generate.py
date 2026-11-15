"""Retrieve + grounded generation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client import QdrantClient

from rag.api.query.service import RetrievedChunk, retrieve
from rag.shared.embeddings import Embedder
from rag.shared.generation import Generator
from rag.shared.types import Timings

SYSTEM_PROMPT = """You are a careful assistant for a RAG system.
Answer the user using ONLY the provided context snippets.
Give a short answer — a few sentences at most.
If the context is insufficient or irrelevant, say you don't know based on the provided sources.
Do not invent facts. When helpful, mention source titles from the context headers."""


@dataclass(frozen=True)
class AskResult:
    query: str
    answer: str
    model: str
    collection: str
    top_k: int
    hits: list[RetrievedChunk]
    timings: Timings


def ask(
    *,
    client: QdrantClient,
    embedder: Embedder,
    generator: Generator,
    collection_name: str,
    query: str,
    top_k: int = 5,
) -> AskResult:
    t_end2end = time.perf_counter()

    retrieved = retrieve(
        client=client,
        embedder=embedder,
        collection_name=collection_name,
        query=query,
        top_k=top_k,
    )
    messages = _build_messages(query=retrieved.query, hits=retrieved.hits)

    t0 = time.perf_counter()
    answer = generator.complete(messages)
    generation_ms = max(0, int(round((time.perf_counter() - t0) * 1000)))

    return AskResult(
        query=retrieved.query,
        answer=answer,
        model=generator.model_id,
        collection=retrieved.collection,
        top_k=retrieved.top_k,
        hits=retrieved.hits,
        timings=Timings(
            end2end_ms=max(0, int(round((time.perf_counter() - t_end2end) * 1000))),
            embedding_ms=retrieved.timings.embedding_ms,
            retrieval_ms=retrieved.timings.retrieval_ms,
            generation_ms=generation_ms,
        ),
    )


def _build_messages(
    *, query: str, hits: list[RetrievedChunk]
) -> list[dict[str, str]]:
    if not hits:
        context = "(no context retrieved)"
    else:
        parts: list[str] = []
        for i, hit in enumerate(hits, start=1):
            title = hit.title or hit.doc_id or "unknown"
            parts.append(
                f"[{i}] title={title}\nsource={hit.source_url or hit.source_path}\n{hit.text}"
            )
        context = "\n\n".join(parts)

    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Give a short answer based only on the context above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
