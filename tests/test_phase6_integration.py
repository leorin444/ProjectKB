"""Phase 6 integration tests: zero-byte file rejection and API robustness."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeEmbeddings, FakeLLM

API_KEY = "test-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        kb_data_dir=tmp_path / "data",
        api_key=API_KEY,
    )
    app = create_app(settings=settings, llm=FakeLLM(), embeddings=FakeEmbeddings())
    with TestClient(app) as c:
        yield c


def test_zero_byte_file_upload_rejected(client: TestClient) -> None:
    # 0-byte document upload must return HTTP 400 Bad Request
    files = {"file": ("empty.txt", b"", "text/plain")}
    resp = client.post("/ingest/file", headers=HEADERS, files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_unsupported_file_extension_rejected(client: TestClient) -> None:
    files = {"file": ("script.exe", b"binarycontent", "application/octet-stream")}
    resp = client.post("/ingest/file", headers=HEADERS, files=files)
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()
