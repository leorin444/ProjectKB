"""Ollama backend implementation for LLM generation, vision, and embeddings."""

from __future__ import annotations

import base64

import httpx

from app.llm.base import LLMError


def model_present(wanted: str, available: list[str]) -> bool:
    """Ollama lists 'name:tag'; a bare name means ':latest'."""
    if ":" not in wanted:
        wanted += ":latest"
    return wanted in available


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str = "llama3.2:1b",
        vision_model: str = "moondream",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.timeout = timeout

    def available_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama is not reachable at {self.base_url} - is it running? "
                f"({type(exc).__name__})"
            ) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def reachable(self) -> None:
        self.available_models()

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"num_ctx": 4096},
        }
        if system:
            body["system"] = system

        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=body,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return str(data.get("response", ""))
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama generate error: {exc}") from exc

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "keep_alive": "30m",
        }
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=body,
                timeout=max(self.timeout, 180.0),
            )
            r.raise_for_status()
            data = r.json()
            return str(data.get("response", ""))
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama vision error: {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            r = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            r.raise_for_status()
            embeddings = r.json().get("embeddings", [])
            return [list(map(float, emb)) for emb in embeddings]
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama embed error: {exc}") from exc
