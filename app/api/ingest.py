"""Ingest API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from app.auth import Principal, get_principal
from app.config import Settings
from app.ingestion.pipeline import (
    detect_document_type,
    ingest_file_bytes,
    ingest_file_job_task,
    sanitize_filename,
)
from app.storage.db import new_id

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_EXTENSIONS = {
    # Documents
    ".pdf", ".txt", ".md", ".csv",
    # Images
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    # Audio
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
    # Video
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
}


@router.post("/file")
async def ingest_file(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File(...)],
    tags: Annotated[str, Form()] = "[]",
    background: Annotated[bool, Form()] = False,
    principal: Principal = Depends(get_principal),
) -> dict:
    settings: Settings = request.app.state.settings
    db = request.app.state.db
    vectors = request.app.state.vectors
    llm = request.app.state.llm
    embeddings = request.app.state.embeddings
    job_manager = request.app.state.job_manager

    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max upload size of {settings.max_upload_mb} MB.",
        )

    try:
        tag_list = json.loads(tags) if tags else []
        if not isinstance(tag_list, list):
            tag_list = [str(tags)]
    except json.JSONDecodeError:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    safe_name = sanitize_filename(filename)
    content_type = file.content_type or "application/octet-stream"
    doc_type = detect_document_type(ext, content_type)

    # Audio/video or explicitly requested background jobs use JobManager
    if background or doc_type in {"audio", "video"}:
        doc_id = new_id()

        settings.originals_dir.mkdir(parents=True, exist_ok=True)
        orig_path = settings.originals_dir / f"{doc_id}{ext}"
        with orig_path.open("wb") as f:
            f.write(file_bytes)

        db.insert_document(
            id=doc_id,
            owner_id=principal.owner_id,
            type=doc_type,
            title=safe_name,
            original_name=safe_name,
            source_path=str(orig_path),
            tags=json.dumps(tag_list),
            status="pending",
        )
        job_id = db.create_job(doc_id)

        job_manager.submit_job(
            job_id,
            ingest_file_job_task,
            job_id=job_id,
            doc_id=doc_id,
            orig_path=orig_path,
            safe_name=safe_name,
            doc_type=doc_type,
            tags=tag_list,
            owner_id=principal.owner_id,
            settings=settings,
            db=db,
            vectors=vectors,
            embeddings=embeddings,
            llm=llm,
        )

        response.status_code = status.HTTP_202_ACCEPTED
        return {
            "job_id": job_id,
            "document_id": doc_id,
            "status": "queued",
            "message": "File accepted for background ingestion",
        }

    # Synchronous processing for documents/images
    result = ingest_file_bytes(
        file_bytes=file_bytes,
        original_name=filename,
        content_type=content_type,
        tags=tag_list,
        owner_id=principal.owner_id,
        settings=settings,
        db=db,
        vectors=vectors,
        embeddings=embeddings,
        llm=llm,
    )

    return result
