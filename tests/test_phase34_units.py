"""Phase 3/4 unit tests: frame deduplication, multimodal time-merging, and job cancellation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.ingestion.media import merge_transcript_and_visuals
from app.ingestion.video_frames import (
    compute_dhash,
    deduplicate_frames,
    hamming_distance,
)
from app.jobs.manager import JobManager
from app.storage.db import SqliteDatabase


def test_dhash_identical_images(tmp_path: Path) -> None:
    img1 = Image.new("RGB", (100, 100), color="white")
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    img1.save(p1)
    img1.save(p2)

    h1 = compute_dhash(p1)
    h2 = compute_dhash(p2)
    assert hamming_distance(h1, h2) == 0


def test_dhash_different_images(tmp_path: Path) -> None:
    # Image 1: left half black, right half white (strong horizontal edges)
    img1 = Image.new("L", (100, 100), 0)
    for x in range(50, 100):
        for y in range(100):
            img1.putpixel((x, y), 255)

    # Image 2: top half black, bottom half white (strong vertical edges, no horizontal ones)
    img2 = Image.new("L", (100, 100), 0)
    for x in range(100):
        for y in range(50, 100):
            img2.putpixel((x, y), 255)

    p1 = tmp_path / "grad1.png"
    p2 = tmp_path / "grad2.png"
    img1.save(p1)
    img2.save(p2)

    h1 = compute_dhash(p1)
    h2 = compute_dhash(p2)
    # Half-black/half-white in different orientations must have different hashes
    assert h1 != h2
    assert hamming_distance(h1, h2) > 6


def test_deduplicate_frames_filters_similar_and_caps(tmp_path: Path) -> None:
    # Create 3 identical frames and 1 distinct frame
    # Use solid white (hash=0) for duplicates; use left-right split for distinct frame
    white_img = Image.new("L", (100, 100), 255)  # uniform → hash=0

    # Distinct: left half black, right half white → non-zero hash
    distinct_img = Image.new("L", (100, 100), 0)
    for x in range(50, 100):
        for y in range(100):
            distinct_img.putpixel((x, y), 255)

    f1 = tmp_path / "f1.jpg"
    f2 = tmp_path / "f2.jpg"
    f3 = tmp_path / "f3.jpg"
    f4 = tmp_path / "f4.jpg"

    white_img.save(f1)
    white_img.save(f2)
    white_img.save(f3)
    distinct_img.save(f4)

    candidates = [(0.0, f1), (10.0, f2), (20.0, f3), (30.0, f4)]
    kept = deduplicate_frames(candidates, threshold=6, max_frames=5)

    # Only 2 distinct frames should be kept (3 identical whites → 1 kept, 1 distinct)
    assert len(kept) == 2
    assert kept[0][0] == 0.0
    assert kept[1][0] == 30.0


def test_merge_transcript_and_visuals() -> None:
    transcript_sections = [
        {
            "heading": "00:00 - 01:00",
            "content": "Discussing the architecture design.",
            "start_ts": 0.0,
            "end_ts": 60.0,
        },
        {
            "heading": "01:00 - 02:00",
            "content": "Demonstrating the terminal error.",
            "start_ts": 60.0,
            "end_ts": 120.0,
        },
    ]

    frame_descriptions = [
        {
            "timestamp": 25.0,
            "frame_path": "/dummy/frame_0001.jpg",
            "description": "Whiteboard diagram showing microservices.",
        },
        {
            "timestamp": 75.0,
            "frame_path": "/dummy/frame_0002.jpg",
            "description": "Terminal showing error: Connection refused on port 8000.",
        },
    ]

    merged = merge_transcript_and_visuals(transcript_sections, frame_descriptions)
    assert len(merged) == 2
    assert "Whiteboard diagram" in merged[0]["content"]
    assert "On-screen (00:25):" in merged[0]["content"]
    assert "Connection refused" in merged[1]["content"]
    assert "On-screen (01:15):" in merged[1]["content"]


def test_job_cancellation(tmp_path: Path) -> None:
    db = SqliteDatabase(tmp_path / "test.db")
    db.init_schema()

    manager = JobManager(db=db, max_workers=1)
    job_id = db.create_job(document_id=None)

    assert not manager.is_cancelled(job_id)
    manager.cancel_job(job_id)
    assert manager.is_cancelled(job_id)

    job_record = db.get_job(job_id)
    assert job_record is not None
    assert job_record["status"] == "cancelled"
    manager.shutdown(wait=True)
