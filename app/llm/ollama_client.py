"""Ollama backend. Phase 0: reachability + model listing only."""

from __future__ import annotations

import httpx

from app.llm.base import LLMError


def model_present(wanted: str, available: list[str]) -> bool:
    """Ollama lists 'name:tag'; a bare name means ':latest'."""
    if ":" not in wanted:
        wanted += ":latest"
    return wanted in available


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama is not reachable at {self.base_url} - is it running? "
                f"({type(exc).__name__})"
            ) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def reachable(self) -> None:
        self.available_models()

    # Implemented in Phase 1 (needs real end-to-end verification before being claimed).
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError("Phase 1")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Phase 1")

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError("Phase 1")
