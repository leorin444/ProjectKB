"""Ingestion pipeline coordinator.

Routes media by type (PDF, image, text/markdown), saves originals under generated IDs,
generates structured Markdown files (source of truth), chunks, embeds, and updates
both SQLite (FTS5 + metadata) and Qdrant.
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
from app.ingestion.pdf import extract_pdf
from app.llm.base import LLMClient
from app.retrieval.embeddings import EmbeddingInterface
from app.storage.db import SqliteDatabase, new_id
from app.storage.vector_store import VectorStore

logger = logging.getLogger("kb.ingestion.pipeline")


def sanitize_filename(filename: str) -> str:
    """Keep basename only, remove risky characters."""
    clean = Path(filename).name
    # Keep alphanumeric, dot, underscore, dash
    clean = "".join(c for c in clean if c.isalnum() or c in "._- ")
    return clean or "unnamed_file"


def detect_document_type(ext: str, content_type: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext == "pdf" or "pdf" in content_type:
        return "pdf"
    if ext in {"png", "jpg", "jpeg", "webp", "bmp"} or "image" in content_type:
        return "image"
    if ext in {"mp4", "mkv", "avi", "mov"} or "video" in content_type:
        return "video"
    return "text"


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
    """Synchronous file ingestion for Phase 1."""
    doc_id = new_id()
    safe_name = sanitize_filename(original_name)
    ext = Path(safe_name).suffix.lower()
    doc_type = detect_document_type(ext, content_type)

    # 1. Save original
    settings.originals_dir.mkdir(parents=True, exist_ok=True)
    orig_path = settings.originals_dir / f"{doc_id}{ext}"
    with orig_path.open("wb") as f:
        f.write(file_bytes)

    # 2. Extract content & sections based on type
    title = Path(safe_name).stem.replace("_", " ").replace("-", " ").title()
    sections: list[dict[str, Any]] = []

    if doc_type == "pdf":
        detected_title, sections = extract_pdf(orig_path)
        if detected_title:
            title = detected_title
    elif doc_type == "image":
        detected_title, sections = extract_image(orig_path, llm)
        if detected_title:
            title = detected_title
    else:
        # Plain text or Markdown
        text = file_bytes.decode("utf-8", errors="replace").strip()
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            title = lines[0].lstrip("# ").strip()
        sections = [{"heading": "Content", "content": text}]

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

    # 5. Chunk the Markdown
    chunks = chunk_markdown(
        md_text,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap,
    )
    if not chunks:
        # Fallback single chunk if empty
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
        "md_path": str(md_path),
    }
