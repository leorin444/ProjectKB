"""Heading-aware Markdown chunker.

Splits on Markdown headings (## / ###) first, preserves code blocks,
prepends the section title to each chunk's text, and parses timestamp boundaries
(e.g. '00:03:12 - 00:05:40') when present.
"""

from __future__ import annotations

import re
from typing import Any

# Match timestamp ranges like "00:03:12 - 00:05:40" or "01:23 - 02:45"
TS_RANGE_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\s*[-–—]\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})"
)


def parse_timestamp_seconds(hours: str | None, minutes: str, seconds: str) -> float:
    h = int(hours) if hours else 0
    m = int(minutes)
    s = int(seconds)
    return float(h * 3600 + m * 60 + s)


def extract_timestamps(heading: str) -> tuple[float | None, float | None]:
    match = TS_RANGE_RE.search(heading)
    if not match:
        return None, None
    g = match.groups()
    start_sec = parse_timestamp_seconds(g[0], g[1], g[2])
    end_sec = parse_timestamp_seconds(g[3], g[4], g[5])
    return start_sec, end_sec


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 chars / token)."""
    return max(1, len(text) // 4)


def split_section_body(
    text: str, max_tokens: int = 500, overlap_tokens: int = 50
) -> list[str]:
    """Split section body into chunks without breaking inside fenced code blocks."""
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    paragraphs = re.split(r"(\n\n+)", text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    inside_code_block = False

    for part in paragraphs:
        code_toggles = part.count("```")
        if code_toggles % 2 != 0:
            inside_code_block = not inside_code_block

        part_tokens = estimate_tokens(part)

        if current_tokens + part_tokens > max_tokens and not inside_code_block and current_parts:
            chunk_str = "".join(current_parts).strip()
            if chunk_str:
                chunks.append(chunk_str)
            # Overlap: keep last part if small
            if overlap_tokens > 0 and current_parts:
                last_part = current_parts[-1]
                if estimate_tokens(last_part) <= overlap_tokens:
                    current_parts = [last_part, part]
                    current_tokens = estimate_tokens(last_part) + part_tokens
                    continue
            current_parts = [part]
            current_tokens = part_tokens
        else:
            current_parts.append(part)
            current_tokens += part_tokens

    if current_parts:
        chunk_str = "".join(current_parts).strip()
        if chunk_str:
            chunks.append(chunk_str)

    # Secondary pass: if any chunk exceeds max_tokens (e.g. single giant paragraph without \n\n),
    # subdivide it by sentences / words to strictly respect token budget
    subdivided_chunks: list[str] = []
    for c in chunks:
        if estimate_tokens(c) <= max_tokens:
            subdivided_chunks.append(c)
        else:
            words = c.split()
            current_words: list[str] = []
            curr_w_tokens = 0
            for w in words:
                w_tok = estimate_tokens(w + " ")
                if curr_w_tokens + w_tok > max_tokens and current_words:
                    subdivided_chunks.append(" ".join(current_words))
                    current_words = [w]
                    curr_w_tokens = w_tok
                else:
                    current_words.append(w)
                    curr_w_tokens += w_tok
            if current_words:
                subdivided_chunks.append(" ".join(current_words))

    return subdivided_chunks


def chunk_markdown(
    markdown_text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Parse Markdown into structured chunks.

    Returns list of dicts with:
    - section: str | None
    - start_ts: float | None
    - end_ts: float | None
    - text: str (section title prepended)
    - token_count: int
    """
    # Strip YAML frontmatter if present
    content = markdown_text
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]

    # Split into sections by Markdown headings (# / ## / ###)
    # Regex splits while capturing heading lines
    heading_pattern = re.compile(r"^(#{1,3}\s+.*)$", re.MULTILINE)
    tokens = heading_pattern.split(content)

    sections: list[tuple[str, str]] = []  # (heading, body)
    current_heading = "General"

    idx = 0
    while idx < len(tokens):
        item = tokens[idx].strip()
        if heading_pattern.match(item):
            # It's a heading
            current_heading = re.sub(r"^#{1,3}\s+", "", item).strip()
            body = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            sections.append((current_heading, body))
            idx += 2
        else:
            if item:
                sections.append((current_heading, item))
            idx += 1

    chunks: list[dict[str, Any]] = []

    for heading, body in sections:
        clean_body = body.strip()
        if not clean_body:
            continue

        start_ts, end_ts = extract_timestamps(heading)
        sub_chunks = split_section_body(clean_body, max_tokens, overlap_tokens)

        for sc in sub_chunks:
            # Prepend section title to chunk text for contextual retrieval
            chunk_text = f"## {heading}\n{sc}" if heading else sc
            chunks.append({
                "section": heading,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "text": chunk_text,
                "token_count": estimate_tokens(chunk_text),
            })

    return chunks
