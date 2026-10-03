"""Dependency health checks for GET /health."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

from pydantic import BaseModel

from app.config import Settings
from app.llm.base import LLMClient, LLMError
from app.llm.ollama_client import model_present
from app.storage.db import SqliteDatabase
from app.storage.vector_store import VectorStore


class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    components: dict[str, ComponentHealth]


def check_ffmpeg() -> ComponentHealth:
    exe = shutil.which("ffmpeg")
    if not exe:
        return ComponentHealth(ok=False, detail="ffmpeg not found on PATH")
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=10, check=True,
            encoding="utf-8", errors="replace",
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        return ComponentHealth(ok=False, detail=f"ffmpeg failed to run: {exc}")
    return ComponentHealth(ok=True, detail=out.splitlines()[0] if out else "ffmpeg ok")


def _guard(fn: Callable[[], str]) -> ComponentHealth:
    try:
        return ComponentHealth(ok=True, detail=fn())
    except Exception as exc:  # noqa: BLE001 - health must never raise
        return ComponentHealth(ok=False, detail=str(exc) or type(exc).__name__)


def check_ollama(llm: LLMClient, settings: Settings) -> dict[str, ComponentHealth]:
    try:
        models = llm.available_models()
    except LLMError as exc:
        down = ComponentHealth(ok=False, detail=str(exc))
        return {"ollama": down}
    result = {"ollama": ComponentHealth(ok=True, detail=f"reachable, {len(models)} model(s)")}
    for key, name in (("llm_model", settings.llm_model), ("vision_model", settings.vision_model)):
        present = model_present(name, models)
        detail = (
            f"{name} available" if present
            else f"model {name} not found - run `ollama pull {name}`"
        )
        result[key] = ComponentHealth(ok=present, detail=detail)
    return result


def build_health(
    settings: Settings, db: SqliteDatabase, vectors: VectorStore, llm: LLMClient
) -> HealthResponse:
    components: dict[str, ComponentHealth] = {}
    components.update(check_ollama(llm, settings))
    if "llm_model" not in components:  # Ollama down: report models explicitly as unknown
        for key, name in (("llm_model", settings.llm_model),
                          ("vision_model", settings.vision_model)):
            components[key] = ComponentHealth(ok=False, detail=f"{name}: cannot check, Ollama down")
    components["database"] = _guard(lambda: (db.ping(), "sqlite ok")[1])
    components["vector_store"] = _guard(lambda: (vectors.health(), "qdrant ok")[1])
    components["ffmpeg"] = check_ffmpeg()
    status = "ok" if all(c.ok for c in components.values()) else "degraded"
    return HealthResponse(status=status, components=components)
