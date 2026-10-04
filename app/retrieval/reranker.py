"""Cross-encoder reranking interface and implementation.

Lazy-loads sentence_transformers.CrossEncoder to ensure fast startup
and fast pytest collection.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger("kb.retrieval.reranker")


class RerankerInterface(Protocol):
    """Protocol for document/chunk rerankers."""

    def rerank(
        self, query: str, items: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]: ...


class CrossEncoderReranker:
    """Local cross-encoder reranker using HuggingFace / sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, items: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        pairs = [[query, item.get("text", "")] for item in items]
        model = self._get_model()
        scores = model.predict(pairs)

        # Attach rerank score and sort descending
        scored_items = []
        for item, score in zip(items, scores, strict=False):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            new_item["score"] = float(score)  # standard key for relevance gating
            scored_items.append(new_item)

        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_items[:top_k]
