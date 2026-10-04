"""Phase 2 unit tests: media time formatting, segment grouping, and JobManager."""

from __future__ import annotations

import time
from pathlib import Path

from app.ingestion.media import format_seconds, group_segments_into_sections
from app.jobs.manager import JobManager
from app.storage.db import SqliteDatabase


def test_format_seconds_zero_and_small() -> None:
    assert format_seconds(0) == "00:00"
    assert format_seconds(45) == "00:45"
    assert format_seconds(65) == "01:05"


def test_format_seconds_hours() -> None:
    assert format_seconds(3600) == "01:00:00"
    assert format_seconds(3665) == "01:01:05"


def test_group_segments_empty() -> None:
    assert group_segments_into_sections([]) == []


def test_group_segments_aggregates_into_intervals() -> None:
    segments = [
        {"start": 0.0, "end": 15.0, "text": "Hello world."},
        {"start": 15.0, "end": 35.0, "text": "This is a podcast."},
        {"start": 35.0, "end": 65.0, "text": "Discussing AI systems."},
        {"start": 65.0, "end": 80.0, "text": "And hardware constraints."},
    ]
    sections = group_segments_into_sections(segments, target_duration=60.0)
    assert len(sections) == 2
    assert sections[0]["start_ts"] == 0.0
    assert sections[0]["end_ts"] == 65.0
    assert "Hello world." in sections[0]["content"]
    assert "Discussing AI systems." in sections[0]["content"]

    assert sections[1]["start_ts"] == 65.0
    assert sections[1]["end_ts"] == 80.0
    assert "hardware constraints" in sections[1]["content"]


def test_job_manager_success_lifecycle(tmp_path: Path) -> None:
    db = SqliteDatabase(tmp_path / "test.db")
    db.init_schema()

    manager = JobManager(db=db, max_workers=1)
    job_id = db.create_job(document_id=None)

    def dummy_task(value: int) -> int:
        time.sleep(0.05)
        return value * 2

    manager.submit_job(job_id, dummy_task, 21)

    # Wait for completion
    for _ in range(20):
        job = db.get_job(job_id)
        if job and job["status"] == "completed":
            break
        time.sleep(0.05)

    final_job = db.get_job(job_id)
    assert final_job is not None
    assert final_job["status"] == "completed"
    assert final_job["progress"] == 100
    assert final_job["stage"] == "done"
    manager.shutdown(wait=True)


def test_job_manager_failure_captures_error(tmp_path: Path) -> None:
    db = SqliteDatabase(tmp_path / "test.db")
    db.init_schema()

    manager = JobManager(db=db, max_workers=1)
    job_id = db.create_job(document_id=None)

    def failing_task() -> None:
        raise ValueError("Simulated media processing error")

    manager.submit_job(job_id, failing_task)

    for _ in range(20):
        job = db.get_job(job_id)
        if job and job["status"] == "failed":
            break
        time.sleep(0.05)

    final_job = db.get_job(job_id)
    assert final_job is not None
    assert final_job["status"] == "failed"
    assert "Simulated media processing error" in str(final_job["error"])
    manager.shutdown(wait=True)
