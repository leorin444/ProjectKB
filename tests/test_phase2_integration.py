"""Phase 2 integration tests: audio/video async ingestion, job polling, and timestamp grounding."""

from __future__ import annotations

import json
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


class FakeLLMPhase2(FakeLLM):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "Based on the recorded audio at [Meeting - 00:01:15 - 00:02:30]: project launched."


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        kb_data_dir=tmp_path / "data",
        api_key=API_KEY,
        min_relevance=0.0,
    )
    app = create_app(settings=settings, llm=FakeLLMPhase2(), embeddings=FakeEmbeddings())
    with TestClient(app) as c:
        yield c


def test_async_audio_ingest_and_job_polling(client) -> None:
    """Simulate audio ingestion via extract_media mock to verify the async pipeline."""
    mock_sections = [
        {
            "heading": "00:00 - 01:15",
            "content": "Introductions and welcoming everyone to the project planning session.",
            "start_ts": 0.0,
            "end_ts": 75.0,
        },
        {
            "heading": "01:15 - 02:30",
            "content": (
                "Discussing the deadline: project launched on October 15 with full offline mode."
            ),
            "start_ts": 75.0,
            "end_ts": 150.0,
        },
    ]

    with patch("app.ingestion.pipeline.extract_media") as mock_extract:
        mock_extract.return_value = ("Team Sync", mock_sections, 150.0)

        # Post dummy mp3 audio file
        files = {"file": ("team_sync.mp3", b"fake mp3 audio bytes", "audio/mpeg")}
        r = client.post(
            "/ingest/file",
            headers=HEADERS,
            files=files,
            data={"tags": json.dumps(["meeting"])},
        )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        job_id = body["job_id"]
        doc_id = body["document_id"]

        # Poll job status until completed (up to 10 seconds)
        for _ in range(100):
            job_r = client.get(f"/jobs/{job_id}", headers=HEADERS)
            assert job_r.status_code == 200
            job_data = job_r.json()
            if job_data["status"] == "completed":
                break
            time.sleep(0.1)

        assert job_data["status"] == "completed"
        assert job_data["progress"] == 100

        # Verify document exists in /documents
        doc = client.get(f"/documents/{doc_id}", headers=HEADERS).json()
        assert doc["type"] == "audio"
        assert "01:15 - 02:30" in doc["markdown"]
        assert "October 15" in doc["markdown"]

        # Verify chunks have start_ts and end_ts
        search_r = client.post(
            "/search", headers=HEADERS, json={"query": "October 15 deadline", "top_k": 5}
        )
        assert search_r.status_code == 200
        hits = search_r.json()
        assert len(hits) >= 1
        top_hit = hits[0]
        assert top_hit["start_ts"] is not None
        assert top_hit["end_ts"] is not None

        # Verify query citations preserve timestamps
        query_r = client.post(
            "/query", headers=HEADERS, json={"question": "When was the project launched?"}
        )
        assert query_r.status_code == 200
        ans = query_r.json()
        assert ans["found"]
        assert len(ans["citations"]) >= 1
        citation = ans["citations"][0]
        assert citation["start_ts"] is not None
        assert citation["end_ts"] is not None


def test_job_not_found(client) -> None:
    r = client.get("/jobs/nonexistent_job", headers=HEADERS)
    assert r.status_code == 404


def test_job_owner_isolation(client, tmp_path: Path) -> None:
    from app.storage.db import SqliteDatabase
    db_path = tmp_path / "data" / "kb.sqlite"
    db = SqliteDatabase(db_path)

    # Insert document belonging to owner 'other'
    doc_id = db.insert_document(owner_id="other", type="text", title="Private Doc")
    job_id = db.create_job(document_id=doc_id)

    # Standard client (owner 'local') should not be able to view user 'other's job
    r = client.get(f"/jobs/{job_id}", headers=HEADERS)
    assert r.status_code == 404
