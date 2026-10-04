"""Jobs API endpoints for polling, cancelling, and re-running background processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth import Principal, get_principal
from app.ingestion.pipeline import ingest_file_job_task

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    id: str
    document_id: str | None = None
    status: str
    stage: str
    progress: int
    error: str | None = None
    created_at: str
    updated_at: str


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    """Poll the status of an asynchronous processing job."""
    db = request.app.state.db
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("document_id"):
        doc = db.get_document(job["document_id"], owner_id=principal.owner_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    """Cancel an active or queued background job."""
    db = request.app.state.db
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("document_id"):
        doc = db.get_document(job["document_id"], owner_id=principal.owner_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Job not found")

    job_manager = request.app.state.job_manager
    job_manager.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@router.post("/{job_id}/rerun")
def rerun_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> dict[str, Any]:
    """Re-run a failed or cancelled processing job."""
    db = request.app.state.db
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    doc_id = job.get("document_id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Job has no associated document to re-run")

    doc = db.get_document(doc_id, owner_id=principal.owner_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Associated document not found")

    source_path = doc.get("source_path")
    if not source_path or not Path(source_path).is_file():
        raise HTTPException(status_code=404, detail="Original source file not found on disk")

    orig_path = Path(source_path)
    tags = json.loads(doc.get("tags") or "[]")
    job_manager = request.app.state.job_manager

    job_manager.submit_job(
        job_id,
        ingest_file_job_task,
        job_id=job_id,
        doc_id=doc_id,
        orig_path=orig_path,
        safe_name=doc.get("original_name") or orig_path.name,
        doc_type=doc.get("type", "text"),
        tags=tags,
        owner_id=principal.owner_id,
        settings=request.app.state.settings,
        db=db,
        vectors=request.app.state.vectors,
        embeddings=request.app.state.embeddings,
        llm=request.app.state.llm,
    )

    return {"job_id": job_id, "status": "queued"}
