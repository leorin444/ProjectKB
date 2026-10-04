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
        reranker=getattr(request.app.state, "reranker", None),
    )
    chunks = retriever.retrieve(
        query=body.question,
        owner_id=principal.owner_id,
        top_k=body.top_k,
    )
    try:
        return generate_grounded_answer(
            question=body.question,
            retrieved_chunks=chunks,
            llm=request.app.state.llm,
            min_relevance=request.app.state.settings.min_relevance,
        )
    except Exception as exc:
        # If the LLM backend (Ollama) is unavailable or errors out,
        # return a helpful 503 Service Unavailable instead of a bare 500.
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation failed: {exc}. Is Ollama running?",
        ) from exc


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
        reranker=getattr(request.app.state, "reranker", None),
    )
    return retriever.retrieve(
        query=body.query,
        owner_id=principal.owner_id,
        top_k=body.top_k,
    )
