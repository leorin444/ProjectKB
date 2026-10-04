"""FastAPI app: wiring and lifespan (init DB + vector store + embeddings + job queue)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.ingest import router as ingest_router
from app.api.jobs import router as jobs_router
from app.api.query import router as query_router
from app.config import Settings, get_settings
from app.health import HealthResponse, build_health
from app.jobs.manager import JobManager
from app.llm.base import LLMClient
from app.llm.ollama_client import OllamaClient
from app.logging_config import log_kv, setup_logging
from app.retrieval.embeddings import LocalEmbeddingModel
from app.retrieval.reranker import CrossEncoderReranker, RerankerInterface
from app.storage.db import SqliteDatabase
from app.storage.vector_store import QdrantEmbeddedStore

logger = logging.getLogger("kb.main")


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embeddings: object | None = None,
    reranker: RerankerInterface | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        settings.ensure_dirs()
        if settings.api_key == "change-me":
            logger.warning("API_KEY is the default 'change-me' - set a real key in .env")

        db = SqliteDatabase(settings.db_path)
        db.init_schema()

        vectors = QdrantEmbeddedStore(settings.index_dir)
        vectors.ensure_collection()

        job_manager = JobManager(db=db, max_workers=1)

        app.state.db = db
        app.state.vectors = vectors
        app.state.settings = settings
        app.state.llm = llm or OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            vision_model=settings.vision_model,
        )
        app.state.embeddings = embeddings or LocalEmbeddingModel(settings.embed_model)
        app.state.job_manager = job_manager
        app.state.reranker = (
            reranker
            if reranker is not None
            else (CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None)
        )

        log_kv(logger, logging.INFO, "startup complete", data_dir=str(settings.data_dir))
        try:
            yield
        finally:
            job_manager.shutdown(wait=False)
            vectors.close()

    app = FastAPI(title="Local Knowledge Base", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        """Report all dependencies. Unauthenticated."""
        s = request.app.state
        return build_health(settings, s.db, s.vectors, s.llm)

    app.include_router(ingest_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(jobs_router)

    return app


app = create_app()
