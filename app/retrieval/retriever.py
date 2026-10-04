"""Hybrid retrieval: Qdrant vector search + SQLite FTS5 keyword search + RRF fusion."""

from __future__ import annotations

from typing import Any

from app.retrieval.embeddings import EmbeddingInterface
from app.retrieval.reranker import RerankerInterface
from app.storage.db import SqliteDatabase
from app.storage.vector_store import VectorStore

RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    fts_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(vector_results, start=1):
        c_id = item["id"]
        doc_map[c_id] = item
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))

    for rank, item in enumerate(fts_results, start=1):
        c_id = item["id"]
        if c_id not in doc_map:
            doc_map[c_id] = item
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))

    # Sort items by descending fused score
    fused: list[dict[str, Any]] = []
    for c_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(doc_map[c_id])
        item["rrf_score"] = score
        # If item doesn't have an explicit score from vector or reranker, expose rrf_score
        if "score" not in item or item["score"] is None:
            item["score"] = score
        fused.append(item)

    return fused


class HybridRetriever:
    def __init__(
        self,
        db: SqliteDatabase,
        vectors: VectorStore,
        embeddings: EmbeddingInterface,
        min_relevance: float = 0.25,
        reranker: RerankerInterface | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embeddings = embeddings
        self.min_relevance = min_relevance
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        owner_id: str,
        top_k: int = 5,
        fetch_limit: int = 20,
    ) -> list[dict[str, Any]]:
        # 1. Vector search
        query_vec = self.embeddings.embed_query(query)
        vec_hits = self.vectors.search_vectors(query_vec, owner_id=owner_id, limit=fetch_limit)

        # 2. Keyword FTS search
        fts_hits = self.db.search_fts(query, owner_id=owner_id, limit=fetch_limit)

        # 3. Fuse with RRF
        merged = reciprocal_rank_fusion(vec_hits, fts_hits)

        # 4. Attach document titles
        results: list[dict[str, Any]] = []
        for item in merged:
            doc_id = item.get("document_id")
            doc = self.db.get_document(doc_id, owner_id=owner_id) if doc_id else None
            title = doc["title"] if doc else "Unknown Document"
            item["document_title"] = title
            results.append(item)

        # 5. Optional Cross-Encoder reranking
        if self.reranker is not None and results:
            return self.reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]
