"""Vector store interface + Qdrant embedded implementation.

Phase 0 covers lifecycle and health only; upsert/search arrive with Phase 1.
Embedded mode locks its folder to ONE process, so only the API process may open it.
A standalone Qdrant server would be a second `VectorStore` implementation selected
by config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION = "chunks"
EMBED_DIM = 384  # all-MiniLM-L6-v2


class VectorStore(Protocol):
    def ensure_collection(self) -> None: ...
    def count(self) -> int: ...
    def health(self) -> None:
        """Raise an exception with a readable message if unusable."""
        ...
    def close(self) -> None: ...


class QdrantEmbeddedStore:
    def __init__(self, path: Path, dim: int = EMBED_DIM) -> None:
        self.path = Path(path)
        self.dim = dim
        self.path.mkdir(parents=True, exist_ok=True)
        # str(): qdrant-client expects a str path. Raises if another process holds the lock.
        self._client = QdrantClient(path=str(self.path))

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
        # Payload indexes (owner_id, document_id) are a no-op in embedded mode; add them
        # in the server implementation.

    def count(self) -> int:
        return int(self._client.count(COLLECTION, exact=True).count)

    def health(self) -> None:
        if not self._client.collection_exists(COLLECTION):
            raise RuntimeError(f"collection '{COLLECTION}' is missing")

    def close(self) -> None:
        self._client.close()
