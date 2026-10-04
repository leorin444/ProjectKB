"""Phase 1 tests: chunker, markdown writer, RRF, and retrieval."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import chunk_markdown, extract_timestamps
from app.ingestion.markdown_writer import format_frontmatter, write_markdown_file
from app.retrieval.answer import (
    NOT_FOUND_MESSAGE,
    generate_grounded_answer,
)
from app.retrieval.retriever import reciprocal_rank_fusion

# ===================== CHUNKER =====================

def test_chunker_splits_on_headings() -> None:
    md = """---
id: abc
---

# Title

## Section One

Content for section one.

## Section Two

Content for section two.
"""
    chunks = chunk_markdown(md)
    sections = [c["section"] for c in chunks]
    assert "Section One" in sections
    assert "Section Two" in sections


def test_chunker_prepends_section_to_text() -> None:
    md = "# Doc\n\n## My Heading\n\nSome content here."
    chunks = chunk_markdown(md)
    assert any("My Heading" in c["text"] for c in chunks)


def test_chunker_preserves_code_block() -> None:
    long_prose = "A " * 300  # ~300 tokens
    code = "```python\nprint('hello')\n```"
    md = f"# Doc\n\n## Section\n\n{long_prose}\n\n{code}"
    chunks = chunk_markdown(md, max_tokens=100)
    full_text = "\n".join(c["text"] for c in chunks)
    assert "```python" in full_text
    assert "print('hello')" in full_text


def test_chunker_extracts_timestamps() -> None:
    start, end = extract_timestamps("00:03:12 - 00:05:40")
    assert start == 3 * 60 + 12
    assert end == 5 * 60 + 40


def test_chunker_extracts_timestamps_with_hours() -> None:
    start, end = extract_timestamps("01:03:00 - 01:05:30")
    assert start == 3600 + 3 * 60
    assert end == 3600 + 5 * 60 + 30


def test_chunker_no_timestamp_in_plain_heading() -> None:
    start, end = extract_timestamps("General Overview")
    assert start is None and end is None


def test_chunker_empty_sections_are_skipped() -> None:
    md = "# Title\n\n## Empty Section\n\n## Real Section\n\nContent."
    chunks = chunk_markdown(md)
    assert all(c["text"].strip() for c in chunks)
    assert any("Real Section" in c["section"] for c in chunks)


# ===================== MARKDOWN WRITER =====================

def test_frontmatter_basic() -> None:
    fm = format_frontmatter({"id": "abc123", "type": "text", "tags": ["t1", "t2"]})
    assert "id: abc123" in fm
    assert 'tags: ["t1", "t2"]' in fm
    assert fm.startswith("---")


def test_frontmatter_skips_none_values() -> None:
    fm = format_frontmatter({"id": "x", "duration": None, "type": "image"})
    assert "duration" not in fm


def test_write_markdown_file(tmp_path: Path) -> None:
    p = tmp_path / "test.md"
    md = write_markdown_file(
        target_path=p,
        metadata={"id": "abc", "type": "text"},
        title="My Test Doc",
        sections=[
            {"heading": "Section One", "content": "Some content."},
            {"heading": "Section Two", "content": "More content."},
        ],
    )
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "# My Test Doc" in text
    assert "## Section One" in text
    assert "## Section Two" in text
    assert "Some content." in text
    assert text == md


def test_write_markdown_file_is_utf8(tmp_path: Path) -> None:
    p = tmp_path / "unicode.md"
    write_markdown_file(
        target_path=p,
        metadata={"id": "u1", "type": "text"},
        title="Unicode: café дані 日本語",
        sections=[{"heading": "Content", "content": "日本語テスト"}],
    )
    text = p.read_text(encoding="utf-8")
    assert "café" in text
    assert "日本語テスト" in text


# ===================== RECIPROCAL RANK FUSION =====================

def test_rrf_empty_both() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_only_vector() -> None:
    v = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    res = reciprocal_rank_fusion(v, [])
    ids = [r["id"] for r in res]
    assert ids == ["a", "b"]


def test_rrf_only_fts() -> None:
    f = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    res = reciprocal_rank_fusion([], f)
    ids = [r["id"] for r in res]
    assert ids == ["a", "b"]


def test_rrf_merges_and_boosts_overlap() -> None:
    v = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    f = [{"id": "b", "text": "y"}, {"id": "c", "text": "z"}]
    res = reciprocal_rank_fusion(v, f)
    ids = [r["id"] for r in res]
    # "b" appears in both lists so should have higher fused score
    assert ids.index("b") < ids.index("c")
    assert all("rrf_score" in r for r in res)


def test_rrf_preserves_all_unique_ids() -> None:
    v = [{"id": str(i), "text": f"v{i}"} for i in range(5)]
    f = [{"id": str(i + 3), "text": f"f{i}"} for i in range(5)]
    res = reciprocal_rank_fusion(v, f)
    ids = {r["id"] for r in res}
    assert ids == {"0", "1", "2", "3", "4", "5", "6", "7"}


# ===================== GROUNDED ANSWER =====================

class FakeLLMYes:
    """TEST DOUBLE: always returns a concrete answer."""
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "The answer is: use --legacy-peer-deps [Doc - Section One]."

    def embed(self, texts):
        raise NotImplementedError

    def describe_image(self, *a):
        raise NotImplementedError


class FakeLLMNotFound:
    """TEST DOUBLE: returns the not-found sentinel."""
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return NOT_FOUND_MESSAGE

    def embed(self, texts):
        raise NotImplementedError

    def describe_image(self, *a):
        raise NotImplementedError


def _chunk(text: str = "npm ERR! peer dependency") -> dict:
    return {
        "id": "c1",
        "document_id": "d1",
        "document_title": "Setup Tutorial",
        "section": "Section One",
        "start_ts": None,
        "end_ts": None,
        "text": text,
        "score": 0.9,
        "rrf_score": 0.03,
    }


def test_grounded_answer_found() -> None:
    result = generate_grounded_answer("fix peer error", [_chunk()], FakeLLMYes())
    assert result.found
    assert "legacy-peer-deps" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "d1"


def test_grounded_answer_not_found_no_chunks() -> None:
    result = generate_grounded_answer("unrelated question", [], FakeLLMYes())
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.citations == []


def test_grounded_answer_not_found_low_score() -> None:
    chunk = {**_chunk(), "score": 0.05, "rrf_score": 0.001}
    result = generate_grounded_answer("question", [chunk], FakeLLMYes(), min_relevance=0.25)
    assert not result.found


def test_grounded_answer_llm_returns_not_found_sentinel() -> None:
    result = generate_grounded_answer("anything", [_chunk()], FakeLLMNotFound())
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
