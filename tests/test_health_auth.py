from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import Principal, get_principal
from app.config import Settings
from app.main import create_app
from tests.conftest import FakeLLM


def _health(settings: Settings, llm: FakeLLM, monkeypatch=None) -> dict:
    with TestClient(create_app(settings, llm=llm)) as client:
        r = client.get("/health")
        assert r.status_code == 200
        return r.json()


def test_health_all_models_present(settings: Settings) -> None:
    body = _health(settings, FakeLLM([settings.llm_model, settings.vision_model]))
    c = body["components"]
    assert c["ollama"]["ok"] and c["llm_model"]["ok"] and c["vision_model"]["ok"]
    assert c["database"]["ok"] and c["vector_store"]["ok"]  # real SQLite + real Qdrant
    assert "ffmpeg" in c


def test_health_missing_model_gives_pull_hint(settings: Settings) -> None:
    body = _health(settings, FakeLLM([settings.llm_model]))
    v = body["components"]["vision_model"]
    assert not v["ok"] and f"ollama pull {settings.vision_model}" in v["detail"]
    assert body["status"] == "degraded"


def test_health_ollama_down_is_reported_not_raised(settings: Settings) -> None:
    body = _health(settings, FakeLLM(down=True))
    c = body["components"]
    assert not c["ollama"]["ok"] and "is it running" in c["ollama"]["detail"]
    assert not c["llm_model"]["ok"]
    assert c["database"]["ok"] and c["vector_store"]["ok"]
    assert body["status"] == "degraded"


def test_health_needs_no_api_key(settings: Settings) -> None:
    with TestClient(create_app(settings, llm=FakeLLM())) as client:
        assert client.get("/health").status_code == 200


def test_cors_off_by_default(settings: Settings) -> None:
    with TestClient(create_app(settings, llm=FakeLLM())) as client:
        r = client.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_api_key_dependency(settings: Settings) -> None:
    app = FastAPI()
    app.state.settings = settings

    @app.get("/secure")
    def secure(p: Principal = Depends(get_principal)) -> dict[str, str]:
        return {"owner": p.owner_id}

    client = TestClient(app)
    assert client.get("/secure").status_code == 401
    assert client.get("/secure", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.get("/secure", headers={"X-API-Key": "test-key"})
    assert ok.status_code == 200 and ok.json() == {"owner": "local"}


def test_ollama_model_matching() -> None:
    from app.llm.ollama_client import model_present

    assert model_present("llama3.1:8b", ["llama3.1:8b"])
    assert model_present("foo", ["foo:latest"])
    assert not model_present("llama3.1:8b", ["llama3.1:70b"])
