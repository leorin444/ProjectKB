"""Markdown writer: builds structured .md files with YAML frontmatter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_frontmatter(metadata: dict[str, Any]) -> str:
    """Format frontmatter cleanly with standard YAML syntax."""
    lines = ["---"]
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, list):
            # Clean list format: [item1, item2]
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        elif isinstance(v, (int, float, bool)):
            lines.append(f"{k}: {str(v).lower() if isinstance(v, bool) else v}")
        else:
            # String value
            s = str(v).strip()
            if any(ch in s for ch in [":", "#", "[", "]", "{", "}", "\n", '"', "'"]):
                lines.append(f"{k}: {json.dumps(s, ensure_ascii=False)}")
            else:
                lines.append(f"{k}: {s}")
    lines.append("---\n")
    return "\n".join(lines)


def write_markdown_file(
    target_path: Path,
    metadata: dict[str, Any],
    title: str,
    sections: list[dict[str, Any]],
) -> str:
    """Write structured Markdown document with frontmatter and sections.

    Each section dict has:
    - 'heading': str (e.g. "Page 1", "00:03:12 - 00:05:40", "Summary", etc.)
    - 'content': str
    """
    front = format_frontmatter(metadata)
    body_parts = [front, f"# {title.strip()}\n"]

    for sec in sections:
        heading = sec.get("heading", "").strip()
        content = sec.get("content", "").strip()
        if heading:
            body_parts.append(f"## {heading}\n")
        if content:
            body_parts.append(f"{content}\n")

    full_text = "\n".join(body_parts).strip() + "\n"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text
