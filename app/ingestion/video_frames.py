"""Video keyframe extraction, perceptual deduplication, and VLM analysis."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from app.ingestion.image import IMAGE_VLM_PROMPT
from app.llm.base import LLMClient

logger = logging.getLogger("kb.ingestion.video_frames")


def compute_dhash(image_path: Path) -> int:
    """Compute 64-bit difference hash (dHash) for an image using Pillow."""
    with Image.open(image_path) as img:
        # Resize to 9x8 grayscale
        resized = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(resized.get_flattened_data())
        diff = 0
        for row in range(8):
            for col in range(8):
                left = pixels[row * 9 + col]
                right = pixels[row * 9 + col + 1]
                diff = (diff << 1) | (1 if left > right else 0)
        return diff


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate Hamming distance between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def extract_candidate_keyframes(
    video_path: Path,
    output_dir: Path,
) -> list[tuple[float, Path]]:
    """Extract candidate keyframes using ffmpeg scene detection + periodic sampling.

    Returns:
    - List of (timestamp_in_seconds, frame_path)
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not found on PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = str(output_dir / "frame_%04d.jpg")

    # Select: scene changes (> 0.3) OR first frame OR
    # fallback sample every ~30s (900 frames at 30fps)
    select_filter = "select='eq(n,0)+gt(scene,0.3)+not(mod(n,900))',showinfo"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        select_filter,
        "-vsync",
        "vfr",
        frame_pattern,
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    # Parse pts_time from ffmpeg showinfo stderr
    pts_times: list[float] = []
    # Lines look like: [Parsed_showinfo_1 ...] n:   0 pts:      0 pts_time:0 ...
    for line in res.stderr.splitlines():
        if "pts_time:" in line:
            match = re.search(r"pts_time:([0-9.]+)", line)
            if match:
                try:
                    pts_times.append(float(match.group(1)))
                except ValueError:
                    pass

    # Match extracted frame files to timestamps
    extracted_files = sorted(output_dir.glob("frame_*.jpg"))
    results: list[tuple[float, Path]] = []

    for idx, frame_file in enumerate(extracted_files):
        ts = pts_times[idx] if idx < len(pts_times) else float(idx * 30.0)
        results.append((ts, frame_file))

    return results


def deduplicate_frames(
    frames: list[tuple[float, Path]],
    threshold: int = 6,
    max_frames: int = 30,
) -> list[tuple[float, Path]]:
    """Filter out near-identical frames using perceptual dHash."""
    if not frames:
        return []

    unique_frames: list[tuple[float, Path]] = []
    seen_hashes: list[int] = []

    for ts, frame_path in frames:
        if len(unique_frames) >= max_frames:
            # Exceeded cap: delete excess frame file
            frame_path.unlink(missing_ok=True)
            continue

        try:
            h = compute_dhash(frame_path)
            # Check against previously kept hashes
            is_duplicate = any(hamming_distance(h, prev) <= threshold for prev in seen_hashes)
            if not is_duplicate:
                unique_frames.append((ts, frame_path))
                seen_hashes.append(h)
            else:
                # Remove duplicate frame file to save disk space
                frame_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Error computing hash for {frame_path}: {exc}")
            unique_frames.append((ts, frame_path))

    return unique_frames


def describe_video_frames(
    frames: list[tuple[float, Path]],
    llm: LLMClient,
) -> list[dict[str, Any]]:
    """Describe each unique keyframe with the vision LLM."""
    results: list[dict[str, Any]] = []

    for ts, frame_path in frames:
        try:
            frame_bytes = frame_path.read_bytes()
            desc = llm.describe_image(frame_bytes, IMAGE_VLM_PROMPT).strip()
            results.append({
                "timestamp": ts,
                "frame_path": str(frame_path),
                "description": desc,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"VLM frame description failed at {ts}s: {exc}")
            results.append({
                "timestamp": ts,
                "frame_path": str(frame_path),
                "description": "(Frame description unavailable)",
            })

    return results
