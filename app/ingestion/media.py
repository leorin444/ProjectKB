"""Audio and video ingestion: ffmpeg audio extraction + faster-whisper transcription."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.ingestion.video_frames import (
    deduplicate_frames,
    describe_video_frames,
    extract_candidate_keyframes,
)
from app.llm.base import LLMClient

logger = logging.getLogger("kb.ingestion.media")


def format_seconds(seconds: float) -> str:
    """Format floating seconds into HH:MM:SS or MM:SS string."""
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_audio_track(media_path: Path, output_wav: Path) -> Path:
    """Extract audio track from media file into 16kHz mono 16-bit PCM WAV via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed or not found on PATH.")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")

    return output_wav


def group_segments_into_sections(
    segments: list[dict[str, Any]],
    target_duration: float = 60.0,
) -> list[dict[str, Any]]:
    """Group fine-grained Whisper segments into coherent ~60s time-coded sections."""
    if not segments:
        return []

    sections: list[dict[str, Any]] = []
    current_texts: list[str] = []
    chunk_start = segments[0]["start"]
    chunk_end = segments[0]["end"]

    for seg in segments:
        current_texts.append(seg["text"].strip())
        chunk_end = seg["end"]

        # If accumulated duration exceeds target, finalize section
        if (chunk_end - chunk_start) >= target_duration:
            start_str = format_seconds(chunk_start)
            end_str = format_seconds(chunk_end)
            sections.append({
                "heading": f"{start_str} - {end_str}",
                "content": " ".join(current_texts).strip(),
                "start_ts": chunk_start,
                "end_ts": chunk_end,
            })
            current_texts = []
            chunk_start = chunk_end

    if current_texts:
        start_str = format_seconds(chunk_start)
        end_str = format_seconds(chunk_end)
        sections.append({
            "heading": f"{start_str} - {end_str}",
            "content": " ".join(current_texts).strip(),
            "start_ts": chunk_start,
            "end_ts": chunk_end,
        })

    return sections


def transcribe_audio(
    wav_path: Path,
    model_size: str = "tiny",
) -> tuple[list[dict[str, Any]], float]:
    """Transcribe 16kHz WAV file using faster-whisper on CPU.

    Returns:
    - segments: list of {"start": float, "end": float, "text": str}
    - duration: total audio duration in seconds
    """
    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper model '{model_size}' on CPU...")
    # CPU int8 quantization is fast and light on RAM
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments_gen, info = model.transcribe(
        str(wav_path),
        beam_size=1,
        word_timestamps=False,
    )

    segments: list[dict[str, Any]] = []
    for s in segments_gen:
        text = s.text.strip()
        if text:
            segments.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": text,
            })

    duration = float(info.duration) if info and info.duration else 0.0
    return segments, duration


def merge_transcript_and_visuals(
    transcript_sections: list[dict[str, Any]],
    frame_descriptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge audio transcript sections with visual keyframe descriptions
    into unified time sections."""
    if not frame_descriptions:
        return transcript_sections

    sections = [dict(s) for s in transcript_sections]

    for frame in frame_descriptions:
        ts = frame["timestamp"]
        desc = frame["description"]
        ts_str = format_seconds(ts)
        visual_block = f"\n\n**On-screen ({ts_str}):**\n{desc}"

        matched = False
        for s in sections:
            start = s.get("start_ts", 0.0)
            end = s.get("end_ts", start + 60.0)
            if start <= ts <= end:
                s["content"] = s["content"] + visual_block
                matched = True
                break

        if not matched:
            sections.append({
                "heading": f"{ts_str} - {format_seconds(ts + 30.0)}",
                "content": f"**On-screen:**\n{desc}",
                "start_ts": ts,
                "end_ts": ts + 30.0,
            })

    sections.sort(key=lambda s: s.get("start_ts", 0.0))
    return sections


def extract_media(
    file_path: Path,
    whisper_model: str = "tiny",
) -> tuple[str, list[dict[str, Any]], float]:
    """Extract audio from audio file and transcribe into time-coded sections."""
    title = file_path.stem.replace("_", " ").replace("-", " ").title()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "extracted.wav"
        extract_audio_track(file_path, wav_path)
        segments, duration = transcribe_audio(wav_path, model_size=whisper_model)
        sections = group_segments_into_sections(segments, target_duration=60.0)

    if not sections:
        sections = [{
            "heading": "00:00 - 00:00",
            "content": "(No spoken words detected in media file)",
            "start_ts": 0.0,
            "end_ts": duration,
        }]

    return title, sections, duration


def extract_video(
    file_path: Path,
    frames_dir: Path,
    llm: LLMClient,
    whisper_model: str = "tiny",
    max_frames: int = 30,
    dedup_threshold: int = 6,
) -> tuple[str, list[dict[str, Any]], float]:
    """Multimodal video extraction: audio transcription + scene keyframes + VLM descriptions."""
    title = file_path.stem.replace("_", " ").replace("-", " ").title()
    sections: list[dict[str, Any]] = []
    duration = 0.0

    # 1. Audio track extraction & transcription (handles silent video gracefully)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "extracted.wav"
            extract_audio_track(file_path, wav_path)
            segments, dur = transcribe_audio(wav_path, model_size=whisper_model)
            duration = dur
            sections = group_segments_into_sections(segments, target_duration=60.0)
    except Exception as exc:  # noqa: BLE001
        logger.info(
            f"Video audio extraction skipped or failed ({exc}); proceeding with visuals only."
        )

    # 2. Keyframe extraction & deduplication
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = extract_candidate_keyframes(file_path, frames_dir)
    kept_frames = deduplicate_frames(raw_frames, threshold=dedup_threshold, max_frames=max_frames)

    # 3. VLM descriptions of unique frames
    frame_descriptions = describe_video_frames(kept_frames, llm)

    # 4. Multimodal time-merging
    merged_sections = merge_transcript_and_visuals(sections, frame_descriptions)
    if not merged_sections:
        merged_sections = [{
            "heading": "00:00 - 00:00",
            "content": "(No spoken or visual content extracted from video)",
            "start_ts": 0.0,
            "end_ts": duration,
        }]

    return title, merged_sections, duration
