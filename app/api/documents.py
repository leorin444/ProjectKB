"""Documents management API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from app.auth import Principal, get_principal
from app.config import Settings
from app.ingestion.chunker import chunk_markdown
from app.storage.db import SqliteDatabase, new_id
from app.storage.vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


class UpdateMarkdownRequest(BaseModel):
    markdown: str


@router.get("")
def list_documents(
    request: Request,
    type: str | None = None,
    tag: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(get_principal),
) -> list[dict[str, Any]]:
    db: SqliteDatabase = request.app.state.db
    with db.connect() as conn:
        query = "SELECT * FROM documents WHERE owner_id=?"
        params: list[Any] = [principal.owner_id]
        if type:
            query += " AND type=?"
            params.append(type)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
        if tag and tag not in d["tags"]:
            continue
        results.append(d)
    return results


@router.get("/{doc_id}")
def get_document(
    request: Request,
    doc_id: str,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    db: SqliteDatabase = request.app.state.db
    doc = db.get_document(doc_id, owner_id=principal.owner_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    doc["tags"] = json.loads(doc["tags"]) if doc.get("tags") else []

    # Read markdown content if available
    md_content = ""
    if doc.get("md_path") and Path(doc["md_path"]).is_file():
        with Path(doc["md_path"]).open(encoding="utf-8") as f:
            md_content = f.read()

    doc["markdown"] = md_content
    return doc


@router.delete("/{doc_id}")
def delete_document(
    request: Request,
    doc_id: str,
    principal: Principal = Depends(get_principal),
) -> dict[str, str]:
    db: SqliteDatabase = request.app.state.db
    vectors: VectorStore = request.app.state.vectors

    doc = db.get_document(doc_id, owner_id=principal.owner_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Delete physical files & frames folder
    for p_key in ("source_path", "md_path"):
        p_val = doc.get(p_key)
        if p_val and Path(p_val).is_file():
            try:
                Path(p_val).unlink()
            except OSError:
                pass

    import shutil
    settings = request.app.state.settings
    doc_frames = settings.frames_dir / doc_id
    if doc_frames.is_dir():
        shutil.rmtree(doc_frames, ignore_errors=True)

    # Delete from Qdrant and SQLite
    vectors.delete_by_document(doc_id, owner_id=principal.owner_id)
    db.delete_document(doc_id, owner_id=principal.owner_id)

    return {"status": "deleted", "id": doc_id}


@router.post("/{doc_id}/reindex")
def reindex_document(
    request: Request,
    doc_id: str,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    db: SqliteDatabase = request.app.state.db
    vectors: VectorStore = request.app.state.vectors
    embeddings = request.app.state.embeddings

    doc = db.get_document(doc_id, owner_id=principal.owner_id)
    if not doc or not doc.get("md_path") or not Path(doc["md_path"]).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document or Markdown source not found.",
        )

    with Path(doc["md_path"]).open(encoding="utf-8") as f:
        md_text = f.read()

    chunks = chunk_markdown(
        md_text,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap,
    )
    chunk_texts = [c["text"] for c in chunks]
    vectors_list = embeddings.embed_texts(chunk_texts)

    db_chunks = []
    vec_chunks = []
    for c, vec in zip(chunks, vectors_list, strict=False):
        c_id = new_id()
        rec = {
            "id": c_id,
            "document_id": doc_id,
            "owner_id": principal.owner_id,
            "section": c.get("section"),
            "start_ts": c.get("start_ts"),
            "end_ts": c.get("end_ts"),
            "text": c["text"],
            "token_count": c.get("token_count", 0),
        }
        db_chunks.append(rec)
        vec_chunks.append({**rec, "vector": vec})

    vectors.delete_by_document(doc_id, owner_id=principal.owner_id)
    db.replace_chunks(doc_id, principal.owner_id, db_chunks)
    vectors.upsert_chunks(vec_chunks)

    return {"status": "reindexed", "document_id": doc_id, "chunk_count": len(db_chunks)}


@router.put("/{doc_id}/markdown")
def update_markdown(
    request: Request,
    doc_id: str,
    body: UpdateMarkdownRequest,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    db: SqliteDatabase = request.app.state.db
    doc = db.get_document(doc_id, owner_id=principal.owner_id)
    if not doc or not doc.get("md_path"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    target_path = Path(doc["md_path"])
    with target_path.open("w", encoding="utf-8") as f:
        f.write(body.markdown)

    return reindex_document(request, doc_id, principal)
