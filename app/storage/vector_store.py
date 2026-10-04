"""Vector store interface + Qdrant embedded implementation.

Embedded mode locks its folder to ONE process, so only the API process may open it.
A standalone Qdrant server would be a second `VectorStore` implementation selected
by config.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION = "chunks"
EMBED_DIM = 384  # all-MiniLM-L6-v2


def _to_uuid_str(val: str) -> str:
    """Ensure val is formatted as a standard 8-4-4-4-12 UUID string."""
    try:
        return str(uuid.UUID(val))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, val))


class VectorStore(Protocol):
    def ensure_collection(self) -> None: ...
    def count(self) -> int: ...
    def health(self) -> None: ...
    def close(self) -> None: ...
    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None: ...
    def search_vectors(
        self, query_vector: list[float], owner_id: str, limit: int = 20
    ) -> list[dict[str, Any]]: ...
    def delete_by_document(self, document_id: str, owner_id: str) -> None: ...


class QdrantEmbeddedStore:
    def __init__(self, path: Path, dim: int = EMBED_DIM) -> None:
        self.path = Path(path)
        self.dim = dim
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.path))

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def count(self) -> int:
        return int(self._client.count(COLLECTION, exact=True).count)

    def health(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            raise RuntimeError(f"collection '{COLLECTION}' is missing")

    def close(self) -> None:
        self._client.close()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        points = []
        for c in chunks:
            point_id = _to_uuid_str(c["id"])
            payload = {
                "chunk_id": c["id"],
                "document_id": c["document_id"],
                "owner_id": c["owner_id"],
                "section": c.get("section"),
                "start_ts": c.get("start_ts"),
                "end_ts": c.get("end_ts"),
                "text": c["text"],
            }
            points.append(PointStruct(id=point_id, vector=c["vector"], payload=payload))
        self._client.upsert(collection_name=COLLECTION, points=points)

    def search_vectors(
        self, query_vector: list[float], owner_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        owner_filter = Filter(
            must=[FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]
        )
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                query_filter=owner_filter,
                limit=limit,
            )
            hits = res.points
        else:
            hits = self._client.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                query_filter=owner_filter,
                limit=limit,
            )
        results = []
        for h in hits:
            payload = h.payload or {}
            results.append({
                "id": payload.get("chunk_id", str(h.id)),
                "document_id": payload.get("document_id"),
                "owner_id": payload.get("owner_id"),
                "section": payload.get("section"),
                "start_ts": payload.get("start_ts"),
                "end_ts": payload.get("end_ts"),
                "text": payload.get("text", ""),
                "score": float(h.score),
            })
        return results

    def delete_by_document(self, document_id: str, owner_id: str) -> None:
        doc_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="owner_id", match=MatchValue(value=owner_id)),
            ]
        )
        self._client.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(filter=doc_filter),
        )
