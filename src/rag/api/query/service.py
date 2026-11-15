"""Embed a query and retrieve top-k chunks from Qdrant."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from rag.shared.embeddings import Embedder
from rag.shared.types import Timings


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    text: str
    chunk_id: str | None
    doc_id: str | None
    chunk_index: int | None
    title: str | None
    source_url: str | None
    source_path: str | None
    filename: str | None


@dataclass(frozen=True)
class RetrieveResult:
    query: str
    collection: str
    top_k: int
    hits: list[RetrievedChunk]
    timings: Timings


def retrieve(
    *,
    client: QdrantClient,
    embedder: Embedder,
    collection_name: str,
    query: str,
    top_k: int = 5,
) -> RetrieveResult:
    query = query.strip()
    if not query:
        raise ValueError("query must be non-empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    t_end2end = time.perf_counter()

    _ensure_qdrant(client)
    _ensure_collection_exists(client, collection_name)

    t0 = time.perf_counter()
    vector = embedder.embed([query])[0]
    embedding_ms = _ms(t0)

    t0 = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    hits = [_to_hit(point) for point in response.points]
    retrieval_ms = _ms(t0)

    return RetrieveResult(
        query=query,
        collection=collection_name,
        top_k=top_k,
        hits=hits,
        timings=Timings(
            end2end_ms=_ms(t_end2end),
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=0,
        ),
    )


def _ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _to_hit(point: Any) -> RetrievedChunk:
    payload = point.payload or {}
    return RetrievedChunk(
        score=float(point.score),
        text=str(payload.get("text") or ""),
        chunk_id=payload.get("chunk_id"),
        doc_id=payload.get("doc_id"),
        chunk_index=payload.get("chunk_index"),
        title=payload.get("title"),
        source_url=payload.get("source_url"),
        source_path=payload.get("source_path"),
        filename=payload.get("filename"),
    )


def _ensure_qdrant(client: QdrantClient) -> None:
    try:
        client.get_collections()
    except UnexpectedResponse as exc:
        raise ConnectionError(f"Qdrant not reachable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ConnectionError(f"Qdrant not reachable: {exc}") from exc


def _ensure_collection_exists(client: QdrantClient, collection_name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        raise FileNotFoundError(
            f"Collection {collection_name!r} not found. Run POST /indexer/run first."
        )
