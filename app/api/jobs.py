"""Jobs API endpoint for polling background processing status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth import Principal, get_principal

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

    # If document_id is present, verify owner isolation
    if job.get("document_id"):
        doc = db.get_document(job["document_id"], owner_id=principal.owner_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Job not found")

    return job
