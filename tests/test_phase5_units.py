"""Phase 5 unit tests: cross-encoder reranker protocol, distractor rejection,
and min_relevance gating.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMClient
from app.retrieval.answer import NOT_FOUND_MESSAGE, generate_grounded_answer
from app.retrieval.reranker import RerankerInterface
from app.retrieval.retriever import HybridRetriever, reciprocal_rank_fusion


class FakeReranker(RerankerInterface):
    """Test double for reranker protocol."""

    def __init__(self, reversed_order: bool = False) -> None:
        self.reversed_order = reversed_order

    def rerank(
        self, query: str, items: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        scored = []
        for i, item in enumerate(items):
            d = dict(item)
            score = float(i) if self.reversed_order else float(len(items) - i)
            d["rerank_score"] = score
            d["score"] = score
            scored.append(d)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class FakeLLMYes(LLMClient):
    def available_models(self) -> list[str]:
        return ["fake"]

    def reachable(self) -> None:
        return

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "The answer is definitely 42 as cited in [Document - Section]."

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384]

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        return "Image"


def test_reciprocal_rank_fusion_attaches_score() -> None:
    vec_results = [{"id": "c1", "text": "vector hit 1"}]
    fts_results = [{"id": "c2", "text": "fts hit 2"}]

    fused = reciprocal_rank_fusion(vec_results, fts_results)
    assert len(fused) == 2
    # Verify both score and rrf_score are attached
    assert "rrf_score" in fused[0]
    assert "score" in fused[0]
    assert fused[0]["score"] == fused[0]["rrf_score"]


def test_hybrid_retriever_applies_reranker() -> None:
    class DummyDB:
        def search_fts(self, query: str, owner_id: str, limit: int = 20) -> list[dict]:
            return [
                {"id": "c1", "document_id": "d1", "text": "Python GIL explanation"},
                {"id": "c2", "document_id": "d1", "text": "SQLite concurrency details"},
            ]

        def get_document(self, doc_id: str, owner_id: str) -> dict:
            return {"title": "Tech Notes"}

    class DummyVectors:
        def search_vectors(
            self, query_vec: list[float], owner_id: str, limit: int = 20
        ) -> list[dict]:
            return []

    class DummyEmbeddings:
        def embed_query(self, text: str) -> list[float]:
            return [0.0] * 384

    retriever = HybridRetriever(
        db=DummyDB(),  # type: ignore[arg-type]
        vectors=DummyVectors(),  # type: ignore[arg-type]
        embeddings=DummyEmbeddings(),  # type: ignore[arg-type]
        reranker=FakeReranker(reversed_order=True),
    )

    hits = retriever.retrieve("concurrency", owner_id="local", top_k=2)
    assert len(hits) == 2
    # FakeReranker with reversed_order gives higher score to later item (c2)
    assert hits[0]["id"] == "c2"
    assert "rerank_score" in hits[0]
    assert hits[0]["score"] == 1.0


def test_unanswerable_query_rejection_below_min_relevance() -> None:
    llm = FakeLLMYes()
    low_relevance_chunks = [
        {
            "id": "c1",
            "document_id": "doc1",
            "document_title": "Cooking Recipes",
            "section": "Pancakes",
            "text": "Mix flour, milk, and eggs into a smooth batter.",
            "score": 0.05,  # lower than min_relevance = 0.25
        }
    ]

    resp = generate_grounded_answer(
        question="What is the quantum spin of a muon?",
        retrieved_chunks=low_relevance_chunks,
        llm=llm,
        min_relevance=0.25,
    )

    assert not resp.found
    assert resp.answer == NOT_FOUND_MESSAGE
    assert len(resp.citations) == 0


def test_answerable_query_accepted_above_min_relevance() -> None:
    llm = FakeLLMYes()
    relevant_chunks = [
        {
            "id": "c1",
            "document_id": "doc1",
            "document_title": "Physics Guide",
            "section": "Muons",
            "text": "A muon has spin 1/2.",
            "score": 0.85,
        }
    ]

    resp = generate_grounded_answer(
        question="What is the quantum spin of a muon?",
        retrieved_chunks=relevant_chunks,
        llm=llm,
        min_relevance=0.25,
    )

    assert resp.found
    assert "42" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].title == "Physics Guide"
