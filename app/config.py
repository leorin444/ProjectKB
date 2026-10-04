"""Application settings. Every path, model name and URL comes from env / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OWNER_ID = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage: everything derives from kb_data_dir unless explicitly overridden.
    kb_data_dir: Path = Path("./data")
    kb_originals_dir: Path | None = None
    kb_frames_dir: Path | None = None

    # Models
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    vision_model: str = "qwen2.5vl:7b"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper_model: str = "small"

    # API / security
    api_key: str = "change-me"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_upload_mb: int = 2048
    cors_origins: str = ""  # comma-separated; empty = CORS disabled
    default_owner_id: str = DEFAULT_OWNER_ID

    # Chunking / retrieval
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    min_relevance: float = 0.25
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Video
    max_frames_per_video: int = 300
    frame_dedup_threshold: int = 6
    section_seconds: int = 120

    log_level: str = "INFO"

    @field_validator("kb_originals_dir", "kb_frames_dir", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        # `.env` lines like `KB_FRAMES_DIR=` arrive as "" and mean "not set".
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def _resolve(self, p: Path) -> Path:
        # Relative paths are anchored at the project root, not the process cwd.
        p = p.expanduser()
        return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.kb_data_dir)

    @property
    def originals_dir(self) -> Path:
        return self._resolve(self.kb_originals_dir) if self.kb_originals_dir else (
            self.data_dir / "originals"
        )

    @property
    def frames_dir(self) -> Path:
        return self._resolve(self.kb_frames_dir) if self.kb_frames_dir else (
            self.data_dir / "frames"
        )

    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "markdown"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "kb.sqlite"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.originals_dir, self.frames_dir, self.markdown_dir,
                  self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
