"""PDF text extraction using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def extract_pdf(file_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Extract text from a PDF file.

    Returns:
    - title: detected document title or file stem
    - sections: list of {"heading": str, "content": str}
    """
    doc = fitz.open(file_path)
    title = doc.metadata.get("title") if doc.metadata else None
    if not title or not title.strip():
        title = file_path.stem.replace("_", " ").replace("-", " ").title()

    sections: list[dict[str, Any]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        heading = f"Page {page_num + 1}"
        sections.append({
            "heading": heading,
            "content": text,
        })

    doc.close()
    return title, sections
