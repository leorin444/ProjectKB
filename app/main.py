"""FastAPI app: wiring and lifespan (init DB + vector store)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.health import HealthResponse, build_health
from app.llm.base import LLMClient
from app.llm.ollama_client import OllamaClient
from app.logging_config import log_kv, setup_logging
from app.storage.db import SqliteDatabase
from app.storage.vector_store import QdrantEmbeddedStore

logger = logging.getLogger("kb.main")


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
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
        app.state.db, app.state.vectors = db, vectors
        app.state.llm = llm or OllamaClient(settings.ollama_base_url)
        log_kv(logger, logging.INFO, "startup complete", data_dir=str(settings.data_dir))
        try:
            yield
        finally:
            vectors.close()  # release the embedded-Qdrant folder lock

    app = FastAPI(title="Local Knowledge Base", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    if settings.cors_origin_list:  # off by default
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origin_list,
            allow_methods=["*"], allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        """Report Ollama, model availability, DB, vector store and ffmpeg. Unauthenticated."""
        s = request.app.state
        return build_health(settings, s.db, s.vectors, s.llm)

    return app


app = create_app()
