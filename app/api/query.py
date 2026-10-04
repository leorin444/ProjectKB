"""Query and search API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.auth import Principal, get_principal
from app.retrieval.answer import AnswerResponse, generate_grounded_answer
from app.retrieval.retriever import HybridRetriever

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@router.post("/query", response_model=AnswerResponse)
def query(
    request: Request,
    body: QueryRequest,
    principal: Principal = Depends(get_principal),
) -> AnswerResponse:
    """Ask a question; returns a grounded answer with citations."""
    retriever = HybridRetriever(
        db=request.app.state.db,
        vectors=request.app.state.vectors,
        embeddings=request.app.state.embeddings,
        min_relevance=request.app.state.settings.min_relevance,
    )
    chunks = retriever.retrieve(
        query=body.question,
        owner_id=principal.owner_id,
        top_k=body.top_k,
    )
    return generate_grounded_answer(
        question=body.question,
        retrieved_chunks=chunks,
        llm=request.app.state.llm,
        min_relevance=request.app.state.settings.min_relevance,
    )


@router.post("/search")
def search(
    request: Request,
    body: SearchRequest,
    principal: Principal = Depends(get_principal),
) -> list[dict[str, Any]]:
    """Return raw retrieved chunks with scores (debugging endpoint)."""
    retriever = HybridRetriever(
        db=request.app.state.db,
        vectors=request.app.state.vectors,
        embeddings=request.app.state.embeddings,
        min_relevance=0.0,
    )
    return retriever.retrieve(
        query=body.query,
        owner_id=principal.owner_id,
        top_k=body.top_k,
    )
