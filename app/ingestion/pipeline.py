"""Ingestion pipeline coordinator.

Routes media by type (PDF, image, audio, video, text/markdown), saves originals
under generated IDs, generates structured Markdown files (source of truth),
chunks, embeds, and updates both SQLite (FTS5 + metadata) and Qdrant.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.ingestion.chunker import chunk_markdown
from app.ingestion.image import extract_image
from app.ingestion.markdown_writer import write_markdown_file
from app.ingestion.media import extract_media
from app.ingestion.pdf import extract_pdf
from app.llm.base import LLMClient
from app.retrieval.embeddings import EmbeddingInterface
from app.storage.db import SqliteDatabase, new_id
from app.storage.vector_store import VectorStore

logger = logging.getLogger("kb.ingestion.pipeline")


def sanitize_filename(filename: str) -> str:
    """Keep basename only, remove risky characters."""
    clean = Path(filename).name
    # Keep alphanumeric, dot, underscore, dash, space
    clean = "".join(c for c in clean if c.isalnum() or c in "._- ")
    return clean or "unnamed_file"


def detect_document_type(ext: str, content_type: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "pdf" or "pdf" in content_type:
        return "pdf"
    if ext in {"png", "jpg", "jpeg", "webp", "bmp"} or "image" in content_type:
        return "image"
    if ext in {"mp3", "wav", "m4a", "ogg", "flac", "aac"} or "audio" in content_type:
        return "audio"
    if ext in {"mp4", "mkv", "avi", "mov", "webm"} or "video" in content_type:
        return "video"
    return "text"


def _process_and_index_document(
    doc_id: str,
    orig_path: Path,
    safe_name: str,
    doc_type: str,
    tags: list[str],
    owner_id: str,
    settings: Settings,
    db: SqliteDatabase,
    vectors: VectorStore,
    embeddings: EmbeddingInterface,
    llm: LLMClient,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Shared document processing: extraction -> Markdown -> chunk -> embed -> index."""
    title = Path(safe_name).stem.replace("_", " ").replace("-", " ").title()
    sections: list[dict[str, Any]] = []
    duration: float | None = None

    if job_id:
        db.update_job(job_id, stage="extracting", progress=20)

    if doc_type == "pdf":
        detected_title, sections = extract_pdf(orig_path)
        if detected_title:
            title = detected_title
    elif doc_type == "image":
        detected_title, sections = extract_image(orig_path, llm)
        if detected_title:
            title = detected_title
    elif doc_type in {"audio", "video"}:
        detected_title, sections, duration = extract_media(
            orig_path, whisper_model=settings.whisper_model
        )
        if detected_title:
            title = detected_title
    else:
        # Plain text or Markdown
        text = orig_path.read_text(encoding="utf-8", errors="replace").strip()
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            title = lines[0].lstrip("# ").strip()
        sections = [{"heading": "Content", "content": text}]

    if job_id:
        db.update_job(job_id, stage="writing_markdown", progress=50)

    # 3. Write structured Markdown file
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    md_path = settings.markdown_dir / f"{doc_id}.md"
    metadata = {
        "id": doc_id,
        "type": doc_type,
        "source": str(orig_path.relative_to(settings.data_dir)).replace("\\", "/"),
        "original_name": safe_name,
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "tags": tags,
    }
    if duration is not None:
        metadata["duration"] = round(duration, 2)

    md_text = write_markdown_file(md_path, metadata, title, sections)

    # 4. Save metadata to SQLite
    db.insert_document(
        id=doc_id,
        owner_id=owner_id,
        type=doc_type,
        title=title,
        original_name=safe_name,
        source_path=str(orig_path),
        md_path=str(md_path),
        tags=json.dumps(tags),
        status="done",
    )

    if job_id:
        db.update_job(job_id, stage="chunking_and_embedding", progress=70)

    # 5. Chunk the Markdown
    chunks = chunk_markdown(
        md_text,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap,
    )
    if not chunks:
        chunks = [{
            "section": "General",
            "start_ts": None,
            "end_ts": None,
            "text": f"## {title}\n(No text content extracted)",
            "token_count": 5,
        }]

    # 6. Embed chunks
    chunk_texts = [c["text"] for c in chunks]
    vectors_list = embeddings.embed_texts(chunk_texts)

    if job_id:
        db.update_job(job_id, stage="indexing", progress=85)

    # Prepare chunks with IDs and vectors
    db_chunks = []
    vector_chunks = []
    for c, vec in zip(chunks, vectors_list, strict=False):
        c_id = new_id()
        chunk_record = {
            "id": c_id,
            "document_id": doc_id,
            "owner_id": owner_id,
            "section": c.get("section"),
            "start_ts": c.get("start_ts"),
            "end_ts": c.get("end_ts"),
            "text": c["text"],
            "token_count": c.get("token_count", 0),
        }
        db_chunks.append(chunk_record)
        vector_chunks.append({**chunk_record, "vector": vec})

    # 7. Index in SQLite and Qdrant
    db.replace_chunks(doc_id, owner_id, db_chunks)
    vectors.upsert_chunks(vector_chunks)

    logger.info(f"Ingested doc={doc_id} type={doc_type} chunks={len(db_chunks)}")
    return {
        "document_id": doc_id,
        "title": title,
        "type": doc_type,
        "chunk_count": len(db_chunks),
        "duration": duration,
    }


def ingest_file_bytes(
    file_bytes: bytes,
    original_name: str,
    content_type: str,
    tags: list[str],
    owner_id: str,
    settings: Settings,
    db: SqliteDatabase,
    vectors: VectorStore,
    embeddings: EmbeddingInterface,
    llm: LLMClient,
) -> dict[str, Any]:
    """Synchronous file ingestion."""
    doc_id = new_id()
    safe_name = sanitize_filename(original_name)
    ext = Path(safe_name).suffix.lower()
    doc_type = detect_document_type(ext, content_type)

    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    orig_path = settings.originals_dir / f"{doc_id}{ext}"
    with orig_path.open("wb") as f:
        f.write(file_bytes)

    return _process_and_index_document(
        doc_id=doc_id,
        orig_path=orig_path,
        safe_name=safe_name,
        doc_type=doc_type,
        tags=tags,
        owner_id=owner_id,
        settings=settings,
        db=db,
        vectors=vectors,
        embeddings=embeddings,
        llm=llm,
    )


def ingest_file_job_task(
    job_id: str,
    doc_id: str,
    orig_path: Path,
    safe_name: str,
    doc_type: str,
    tags: list[str],
    owner_id: str,
    settings: Settings,
    db: SqliteDatabase,
    vectors: VectorStore,
    embeddings: EmbeddingInterface,
    llm: LLMClient,
) -> None:
    """Target function executed by the JobManager background worker."""
    _process_and_index_document(
        doc_id=doc_id,
        orig_path=orig_path,
        safe_name=safe_name,
        doc_type=doc_type,
        tags=tags,
        owner_id=owner_id,
        settings=settings,
        db=db,
        vectors=vectors,
        embeddings=embeddings,
        llm=llm,
        job_id=job_id,
    )
