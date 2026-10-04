"""Test doubles are clearly labelled `Fake*`; everything else is real."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # _env_file=None: tests must never read the developer's real .env
    return Settings(_env_file=None, kb_data_dir=tmp_path / "kb data", api_key="test-key")


class FakeLLM:
    """TEST DOUBLE for the Ollama backend (health checks only)."""

    def __init__(self, models: list[str] | None = None, down: bool = False) -> None:
        self.models = models or []
        self.down = down

    def available_models(self) -> list[str]:
        from app.llm.base import LLMError

        if self.down:
            raise LLMError("Ollama is not reachable at http://test - is it running?")
        return self.models

    def reachable(self) -> None:
        self.available_models()

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError


class FakeEmbeddings:
    """TEST DOUBLE: deterministic unit-length 384-dim embeddings (no model download)."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        out = []
        for t in texts:
            digest = hashlib.md5(t.encode(), usedforsecurity=False).digest()
            base = [(b / 255.0 - 0.5) for b in digest]
            v = (base * (384 // len(base) + 1))[:384]
            norm = sum(x ** 2 for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
