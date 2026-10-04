"""Ingest API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.auth import Principal, get_principal
from app.config import Settings
from app.ingestion.pipeline import ingest_file_bytes

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt", ".md", ".csv"
}


@router.post("/file")
async def ingest_file(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    tags: Annotated[str, Form()] = "[]",
    principal: Principal = Depends(get_principal),
) -> dict:
    settings: Settings = request.app.state.settings
    db = request.app.state.db
    vectors = request.app.state.vectors
    llm = request.app.state.llm
    embeddings = request.app.state.embeddings

    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
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

    result = ingest_file_bytes(
        file_bytes=file_bytes,
        original_name=filename,
        content_type=file.content_type or "application/octet-stream",
        tags=tag_list,
        owner_id=principal.owner_id,
        settings=settings,
        db=db,
        vectors=vectors,
        embeddings=embeddings,
        llm=llm,
    )

    return result
