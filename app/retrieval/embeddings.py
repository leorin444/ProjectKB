"""Embedding interface and SentenceTransformers implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST


class EmbeddingInterface(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbeddingModel:
    """Wraps SentenceTransformer with lazy loading — model loads on first use,
    not at construction time, so the FastAPI lifespan completes instantly."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: _ST | None = None

    def _get_model(self) -> _ST:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._get_model().encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
