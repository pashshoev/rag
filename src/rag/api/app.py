"""FastAPI application — mounts feature routers (indexer, …)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from qdrant_client import QdrantClient

from rag.api.indexer import router as indexer_router
from rag.api.query import router as query_router
from rag.api.settings import load_settings
from rag.shared.embeddings import create_embedder
from rag.shared.generation import create_generator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    app.state.settings = settings
    app.state.embedder = create_embedder(
        backend=settings.embedding_backend,
        model_id=settings.embedding_model,
        base_url=settings.embedding_base_url,
    )
    app.state.generator = create_generator(
        base_url=settings.llm_base_url,
        model_id=settings.llm_model,
        api_key=settings.llm_api_key,
    )
    app.state.qdrant = QdrantClient(url=settings.qdrant_url)
    yield
    app.state.qdrant.close()


def create_app() -> FastAPI:
    app = FastAPI(title="rag", version="0.1.0", lifespan=lifespan)
    app.include_router(indexer_router)
    app.include_router(query_router)
    return app


app = create_app()
