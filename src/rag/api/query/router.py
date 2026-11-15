"""Retrieval + grounded generation routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rag.api.query.generate import ask
from rag.api.query.service import retrieve
from rag.shared.types import Timings

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question / search text")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to return")


class RetrievedChunkResponse(BaseModel):
    score: float
    text: str
    chunk_id: str | None = None
    doc_id: str | None = None
    chunk_index: int | None = None
    title: str | None = None
    source_url: str | None = None
    source_path: str | None = None
    filename: str | None = None


class QueryResponse(BaseModel):
    query: str
    collection: str
    top_k: int
    hits: list[RetrievedChunkResponse]
    timings: Timings


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    collection: str
    top_k: int
    citations: list[RetrievedChunkResponse]
    timings: Timings


@router.post("/retrieve", response_model=QueryResponse)
def retrieve_chunks(body: QueryRequest, request: Request) -> QueryResponse:
    """Embed the query and return top-k similar chunks (retrieval only)."""
    settings = request.app.state.settings
    try:
        result = retrieve(
            client=request.app.state.qdrant,
            embedder=request.app.state.embedder,
            collection_name=settings.collection_name,
            query=body.query,
            top_k=body.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryResponse(
        query=result.query,
        collection=result.collection,
        top_k=result.top_k,
        hits=[RetrievedChunkResponse(**hit.__dict__) for hit in result.hits],
        timings=result.timings,
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(body: QueryRequest, request: Request) -> AskResponse:
    """Retrieve top-k chunks, then generate a grounded answer via the LLM."""
    settings = request.app.state.settings
    try:
        result = ask(
            client=request.app.state.qdrant,
            embedder=request.app.state.embedder,
            generator=request.app.state.generator,
            collection_name=settings.collection_name,
            query=body.query,
            top_k=body.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        query=result.query,
        answer=result.answer,
        model=result.model,
        collection=result.collection,
        top_k=result.top_k,
        citations=[RetrievedChunkResponse(**hit.__dict__) for hit in result.hits],
        timings=result.timings,
    )
