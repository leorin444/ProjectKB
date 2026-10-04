"""Background job manager using a single-worker queue to protect CPU and RAM."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.storage.db import SqliteDatabase

logger = logging.getLogger("kb.jobs")


class JobCancelledError(Exception):
    """Raised when an in-progress job is cancelled."""


class JobManager:
    """Manages asynchronous background processing with persistent SQLite status."""

    def __init__(self, db: SqliteDatabase, max_workers: int = 1) -> None:
        self.db = db
        # Single worker ensures heavy Whisper or VLM inferences do not run concurrently
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kb_worker")
        self._lock = threading.Lock()
        self._cancelled_jobs: set[str] = set()

    def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled."""
        with self._lock:
            self._cancelled_jobs.add(job_id)
            self.db.update_job(job_id, status="cancelled", stage="cancelled")
        logger.info(f"Job {job_id} cancelled by user")
        return True

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been requested to cancel."""
        with self._lock:
            return job_id in self._cancelled_jobs

    def submit_job(
        self,
        job_id: str,
        task_fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Submit a task to be processed in the background."""
        with self._lock:
            self._cancelled_jobs.discard(job_id)
        self.db.update_job(job_id, status="queued", stage="queued", progress=0)

        def _runner() -> None:
            if self.is_cancelled(job_id):
                logger.info(f"Job {job_id} skipped because it was cancelled")
                return

            try:
                self.db.update_job(job_id, status="processing", stage="starting", progress=5)
                logger.info(f"Job {job_id} started processing")
                task_fn(*args, **kwargs)
                if not self.is_cancelled(job_id):
                    self.db.update_job(job_id, status="completed", stage="done", progress=100)
                    logger.info(f"Job {job_id} completed successfully")
            except JobCancelledError:
                logger.info(f"Job {job_id} was aborted due to cancellation")
                self.db.update_job(job_id, status="cancelled", stage="cancelled")
            except Exception as exc:
                if self.is_cancelled(job_id):
                    self.db.update_job(job_id, status="cancelled", stage="cancelled")
                    return
                logger.exception(f"Job {job_id} failed with error: {exc}")
                self.db.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    error=str(exc),
                )

        self._executor.submit(_runner)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
