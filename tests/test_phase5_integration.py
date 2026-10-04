"""Phase 5 integration tests: end-to-end evaluation of unanswerable distractors
and cross-encoder reranker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.retrieval.reranker import RerankerInterface
from tests.conftest import FakeEmbeddings, FakeLLM

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


class FakeReranker(RerankerInterface):
    """Test double for reranker protocol."""

    def __init__(self, reversed_order: bool = True) -> None:
        self.reversed_order = reversed_order

    def rerank(
        self, query: str, items: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        scored = []
        for i, item in enumerate(items):
            d = dict(item)
            score = float(i + 1) if self.reversed_order else float(len(items) - i)
            d["rerank_score"] = score
            d["score"] = score
            scored.append(d)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class FakeLLMPhase5(FakeLLM):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return (
            "Docker container isolation uses Linux namespaces and cgroups "
            "[Docker Architecture - Namespaces]."
        )


@pytest.fixture
def app_with_reranker(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        kb_data_dir=tmp_path / "data",
        api_key=API_KEY,
        use_reranker=True,
        min_relevance=0.25,
    )
    return create_app(
        settings=settings,
        llm=FakeLLMPhase5(),
        embeddings=FakeEmbeddings(),
        reranker=FakeReranker(reversed_order=True),
    )


def test_reranker_reorders_search_results(app_with_reranker) -> None:
    with TestClient(app_with_reranker) as client:
        # Ingest document with two distinct sections
        doc_content = (
            "# Docker Architecture\n\n"
            "## Storage\nDocker storage drivers include overlay2 and btrfs.\n\n"
            "## Namespaces\nDocker container isolation uses Linux namespaces and cgroups.\n"
        )
        files = {"file": ("docker.md", doc_content.encode("utf-8"), "text/markdown")}
        r = client.post("/ingest/file", headers=HEADERS, files=files)
        assert r.status_code == 200

        # Run /search endpoint with reranker active
        search_res = client.post(
            "/search", headers=HEADERS, json={"query": "Docker container isolation", "top_k": 2}
        )
        assert search_res.status_code == 200
        hits = search_res.json()
        assert len(hits) == 2
        # Verify rerank_score is present on results
        assert "rerank_score" in hits[0]
        assert hits[0]["rerank_score"] >= hits[1]["rerank_score"]


def test_unanswerable_distractor_query_returns_not_found(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        kb_data_dir=tmp_path / "data",
        api_key=API_KEY,
        min_relevance=0.80,  # High threshold to reject irrelevant distractors
    )
    app = create_app(settings=settings, llm=FakeLLMPhase5(), embeddings=FakeEmbeddings())

    with TestClient(app) as client:
        # Ingest an astronomy article
        doc_content = (
            "# Jupiter and its Moons\n\n"
            "## Ganymede\n"
            "Ganymede is the largest moon in the Solar System, even bigger than Mercury.\n"
        )
        files = {"file": ("jupiter.md", doc_content.encode("utf-8"), "text/markdown")}
        r = client.post("/ingest/file", headers=HEADERS, files=files)
        assert r.status_code == 200

        # Ask an unrelated distractor query that has no match in the KB
        resp = client.post(
            "/query",
            headers=HEADERS,
            json={"question": "What is the optimal baking temperature for sourdough bread?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert not data["found"]
        assert "Not found in the knowledge base." in data["answer"]
        assert len(data["citations"]) == 0
