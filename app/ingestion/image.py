"""Image and screenshot ingestion: VLM description with optional OCR fallback."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from app.llm.base import LLMClient, LLMError

logger = logging.getLogger("kb.ingestion.image")

IMAGE_VLM_PROMPT = (
    "Provide a structured analysis of this image/screenshot:\n"
    "1. A short summary description of what is shown.\n"
    "2. Transcribe all visible text faithfully. Do not invent or assume text that is not visible.\n"
    "3. If any tables are visible, format them as Markdown tables.\n"
    "4. If any code or terminal output is visible, wrap it in fenced code blocks with "
    "appropriate language tags.\n"
    "5. Include UI context: application name, window titles, buttons, and error messages."
)


def _try_tesseract_ocr(image_path: Path) -> str | None:
    """Optional OCR fallback using pytesseract if tesseract is on PATH."""
    try:
        import pytesseract

        if not shutil.which("tesseract"):
            return None
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img).strip()
            return text if text else None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Tesseract OCR fallback skipped: {exc}")
        return None


def extract_image(
    image_path: Path,
    llm: LLMClient,
) -> tuple[str, list[dict[str, Any]]]:
    """Analyze image with VLM (or OCR fallback).

    Returns:
    - title: image title
    - sections: list of {"heading": str, "content": str}
    """
    title = image_path.stem.replace("_", " ").replace("-", " ").title()
    content = ""

    with image_path.open("rb") as f:
        image_bytes = f.read()

    try:
        content = llm.describe_image(image_bytes, IMAGE_VLM_PROMPT).strip()
    except (LLMError, NotImplementedError, Exception) as exc:  # noqa: BLE001
        logger.warning(f"VLM describe_image failed for {image_path.name}: {exc}")

    if not content:
        # Try OCR fallback
        ocr_text = _try_tesseract_ocr(image_path)
        if ocr_text:
            content = f"**Transcribed Text (OCR):**\n\n```\n{ocr_text}\n```"
        else:
            content = f"Screenshot image: {image_path.name} (visual details pending VLM review)."

    sections = [
        {"heading": "Visual Description & Content", "content": content}
    ]
    return title, sections
