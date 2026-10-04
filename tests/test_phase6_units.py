"""Phase 6 unit tests: input validation, zero-byte file handling, and chunker edge cases."""

from __future__ import annotations

from app.ingestion.chunker import chunk_markdown


def test_chunk_markdown_empty_or_whitespace() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  \t  ") == []


def test_chunk_markdown_frontmatter_only() -> None:
    text = "---\ntitle: Only Frontmatter\ndate: 2026-10-04\n---\n"
    assert chunk_markdown(text) == []


def test_chunk_markdown_single_heading_no_body() -> None:
    text = "## Empty Section With No Text"
    assert chunk_markdown(text) == []


def test_chunk_markdown_huge_single_paragraph() -> None:
    # A single continuous sentence longer than max_tokens
    words = ["word" for _ in range(300)]
    text = f"## Long Section\n{' '.join(words)}"
    chunks = chunk_markdown(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) > 1
    # Check that each sub-chunk has the heading attached
    for c in chunks:
        assert c["section"] == "Long Section"
        assert c["text"].startswith("## Long Section")
