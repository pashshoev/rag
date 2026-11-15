"""Index corpus → chunk → embed → Qdrant."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.shared import CharacterChunker, TextFolderSource
from rag.shared.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexResult:
    collection: str
    docs: int
    chunks: int
    points: int
    recreate: bool


def index_corpus(
    *,
    client: QdrantClient,
    embedder: Embedder,
    corpus_dir: Path,
    collection_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    recreate: bool = True,
) -> IndexResult:
    """Load texts, chunk, embed, and upsert into Qdrant.

    ``recreate=True`` (default): delete the collection if it exists, then create
    fresh — avoids duplicate pollution on re-runs.

    ``recreate=False``: ensure collection exists and upsert by deterministic
    point ids (stable per chunk id).
    """
    _ensure_qdrant(client)

    source = TextFolderSource(corpus_dir)
    chunker = CharacterChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    docs = list(source.iter_documents())
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.split(doc))

    if not all_chunks:
        raise ValueError(f"No chunks produced from {corpus_dir}")

    _ensure_collection(
        client,
        collection_name=collection_name,
        dimension=embedder.dimension,
        recreate=recreate,
    )

    vectors = embedder.embed([c.text for c in all_chunks])
    points = [
        PointStruct(
            id=_point_id(chunk.id),
            vector=vector,
            payload={
                "chunk_id": chunk.id,
                "text": chunk.text,
                "doc_id": chunk.metadata.get("doc_id"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "source_path": chunk.metadata.get("source_path"),
                "source_url": chunk.metadata.get("source_url"),
                "title": chunk.metadata.get("title"),
                "filename": chunk.metadata.get("filename"),
            },
        )
        for chunk, vector in zip(all_chunks, vectors, strict=True)
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info(
        "indexed collection=%s docs=%s chunks=%s recreate=%s",
        collection_name,
        len(docs),
        len(points),
        recreate,
    )
    return IndexResult(
        collection=collection_name,
        docs=len(docs),
        chunks=len(all_chunks),
        points=len(points),
        recreate=recreate,
    )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _ensure_qdrant(client: QdrantClient) -> None:
    try:
        client.get_collections()
    except UnexpectedResponse as exc:
        raise ConnectionError(f"Qdrant not reachable: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — surface any transport failure clearly
        raise ConnectionError(f"Qdrant not reachable: {exc}") from exc


def _ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    dimension: int,
    recreate: bool,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if recreate and collection_name in existing:
        client.delete_collection(collection_name)
        existing.discard(collection_name)

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
