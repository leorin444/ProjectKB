"""Background job manager using a single-worker queue to protect CPU and RAM."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.storage.db import SqliteDatabase

logger = logging.getLogger("kb.jobs")


class JobManager:
    """Manages asynchronous background processing with persistent SQLite status."""

    def __init__(self, db: SqliteDatabase, max_workers: int = 1) -> None:
        self.db = db
        # Single worker ensures heavy Whisper or VLM inferences do not run concurrently
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kb_worker")
        self._lock = threading.Lock()

    def submit_job(
        self,
        job_id: str,
        task_fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Submit a task to be processed in the background."""
        self.db.update_job(job_id, status="queued", stage="queued", progress=0)

        def _runner() -> None:
            try:
                self.db.update_job(job_id, status="processing", stage="starting", progress=5)
                logger.info(f"Job {job_id} started processing")
                task_fn(*args, **kwargs)
                self.db.update_job(job_id, status="completed", stage="done", progress=100)
                logger.info(f"Job {job_id} completed successfully")
            except Exception as exc:
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
