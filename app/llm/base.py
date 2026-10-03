"""Model access contract. All model use goes through this Protocol.

A cloud provider = a new class implementing it + a config switch.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Readable, actionable error from a model backend."""


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def describe_image(self, image_bytes: bytes, prompt: str) -> str: ...

    def reachable(self) -> None:
        """Raise LLMError with a readable message if the backend cannot be reached."""
        ...

    def available_models(self) -> list[str]: ...
