"""Phase 3/4 integration tests: video keyframe ingestion, job cancel/rerun, and visual RAG."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeEmbeddings, FakeLLM

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


class FakeLLMPhase34(FakeLLM):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return (
            "Based on the video recording at [Tutorial - 00:01:10 - 00:02:20]: "
            "the terminal showed error code ERESOLVE peer dependency."
        )


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        kb_data_dir=tmp_path / "data",
        api_key=API_KEY,
        min_relevance=-1.0,  # FakeEmbeddings produce arbitrary cosine scores; bypass score gate
    )
    app = create_app(settings=settings, llm=FakeLLMPhase34(), embeddings=FakeEmbeddings())
    with TestClient(app) as c:
        yield c


def test_job_cancel_and_rerun_endpoints(client) -> None:
    # Ingest a text file with background=True to produce a queued job
    files = {"file": ("notes.txt", b"Some pending notes", "text/plain")}
    r = client.post("/ingest/file", headers=HEADERS, files=files, data={"background": "true"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # Test cancel endpoint
    cancel_r = client.post(f"/jobs/{job_id}/cancel", headers=HEADERS)
    assert cancel_r.status_code == 200
    assert cancel_r.json()["status"] == "cancelled"

    # Test re-run endpoint
    rerun_r = client.post(f"/jobs/{job_id}/rerun", headers=HEADERS)
    assert rerun_r.status_code == 200
    assert rerun_r.json()["status"] == "queued"

    # Wait for completion of re-run
    for _ in range(50):
        info = client.get(f"/jobs/{job_id}", headers=HEADERS).json()
        if info["status"] == "completed":
            break
        time.sleep(0.1)

    assert info["status"] == "completed"


def test_video_keyframe_ingestion_and_visual_query(client) -> None:
    mock_merged_sections = [
        {
            "heading": "00:00 - 01:10",
            "content": (
                "Spoken introduction to setting up the environment.\n\n"
                "**On-screen (00:00:30):**\n"
                "Visual slide titled Environment Setup Overview."
            ),
            "start_ts": 0.0,
            "end_ts": 70.0,
        },
        {
            "heading": "01:10 - 02:20",
            "content": (
                "Spoken explanation: the package installation encountered an issue.\n\n"
                "**On-screen (00:01:45):**\n"
                "Terminal showing error code ERESOLVE peer dependency failed to resolve."
            ),
            "start_ts": 70.0,
            "end_ts": 140.0,
        },
    ]

    with patch("app.ingestion.pipeline.extract_video") as mock_extract_video:
        mock_extract_video.return_value = ("Tutorial", mock_merged_sections, 140.0)

        # Upload MP4 video
        files = {"file": ("tutorial.mp4", b"fake mp4 video bytes", "video/mp4")}
        r = client.post("/ingest/file", headers=HEADERS, files=files)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        doc_id = r.json()["document_id"]

        # Wait for processing
        for _ in range(50):
            job_info = client.get(f"/jobs/{job_id}", headers=HEADERS).json()
            if job_info["status"] == "completed":
                break
            time.sleep(0.1)

        assert job_info["status"] == "completed"

        # Verify document Markdown contains both spoken and on-screen content
        doc = client.get(f"/documents/{doc_id}", headers=HEADERS).json()
        assert doc["type"] == "video"
        assert "Environment Setup Overview" in doc["markdown"]
        assert "ERESOLVE peer dependency" in doc["markdown"]

        # Verify search retrieves on-screen keyframe text
        search_r = client.post(
            "/search", headers=HEADERS, json={"query": "ERESOLVE peer dependency", "top_k": 5}
        )
        assert search_r.status_code == 200
        hits = search_r.json()
        assert len(hits) >= 1
        assert "On-screen (00:01:45):" in hits[0]["text"]
        assert hits[0]["start_ts"] == 70.0
        assert hits[0]["end_ts"] == 140.0

        # Verify query with citation grounded in visual content
        query_r = client.post(
            "/query", headers=HEADERS, json={"question": "What error appeared on the screen?"}
        )
        assert query_r.status_code == 200
        ans = query_r.json()
        assert ans["found"]
        assert len(ans["citations"]) >= 1
        top_citation = ans["citations"][0]
        assert top_citation["start_ts"] == 70.0
        assert top_citation["end_ts"] == 140.0
