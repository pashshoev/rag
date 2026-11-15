"""Indexer HTTP routes — kept separate from query/generation routers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rag.api.indexer.service import index_corpus
from rag.api.settings import Settings

router = APIRouter(prefix="/indexer", tags=["indexer"])


class IndexRequest(BaseModel):
    recreate: bool = Field(
        default=True,
        description=(
            "If true, drop and recreate the collection before indexing "
            "(safe default for re-runs). If false, upsert by stable point ids."
        ),
    )


class IndexResponse(BaseModel):
    collection: str
    docs: int
    chunks: int
    points: int
    recreate: bool


@router.post("/run", response_model=IndexResponse)
def run_index(body: IndexRequest, request: Request) -> IndexResponse:
    """Chunk + embed corpus into Qdrant. Can be slow; runs in the API process."""
    settings: Settings = request.app.state.settings
    try:
        result = index_corpus(
            client=request.app.state.qdrant,
            embedder=request.app.state.embedder,
            corpus_dir=settings.corpus_dir,
            collection_name=settings.collection_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            recreate=body.recreate,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IndexResponse(
        collection=result.collection,
        docs=result.docs,
        chunks=result.chunks,
        points=result.points,
        recreate=result.recreate,
    )
