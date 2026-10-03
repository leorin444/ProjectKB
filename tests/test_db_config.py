from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.storage.db import SqliteDatabase, fts_match_query


def test_fts5_available_in_this_python_build() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.execute("INSERT INTO t(x) VALUES ('hello world')")
        assert conn.execute("SELECT count(*) FROM t WHERE t MATCH 'hello'").fetchone()[0] == 1
    except sqlite3.OperationalError as exc:  # pragma: no cover
        pytest.fail(f"FTS5 not available: {exc}")
    finally:
        conn.close()


@pytest.fixture
def db(tmp_path: Path) -> SqliteDatabase:
    d = SqliteDatabase(tmp_path / "kb.sqlite")
    d.init_schema()
    return d


def _doc(db: SqliteDatabase, owner: str = "local") -> str:
    return db.insert_document(owner_id=owner, type="text", title="T")


def test_init_schema_is_idempotent(db: SqliteDatabase) -> None:
    db.init_schema()
    db.init_schema()
    with db.connect() as c:
        assert c.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 1


def test_fts_sync_insert_update_delete(db: SqliteDatabase) -> None:
    doc = _doc(db)
    db.replace_chunks(doc, "local", [{"id": "c1", "text": "peer dependency error"}])
    assert [r["id"] for r in db.search_fts("dependency", "local")] == ["c1"]

    with db.connect() as c:
        c.execute("UPDATE chunks SET text='something else' WHERE id='c1'")
    assert db.search_fts("dependency", "local") == []
    assert [r["id"] for r in db.search_fts("something", "local")] == ["c1"]

    db.delete_document(doc, "local")  # cascades to chunks and FTS
    assert db.search_fts("something", "local") == []
    assert db.count_chunks() == 0


def test_replace_chunks_is_idempotent(db: SqliteDatabase) -> None:
    doc = _doc(db)
    for _ in range(3):
        db.replace_chunks(doc, "local", [{"text": "alpha"}, {"text": "beta"}])
    assert db.count_chunks(doc) == 2
    assert len(db.search_fts("alpha", "local")) == 1


def test_fts_search_is_owner_scoped(db: SqliteDatabase) -> None:
    a, b = _doc(db, "alice"), _doc(db, "bob")
    db.replace_chunks(a, "alice", [{"text": "secret alpha"}])
    db.replace_chunks(b, "bob", [{"text": "secret beta"}])
    assert len(db.search_fts("secret", "alice")) == 1
    assert db.search_fts("secret", "alice")[0]["document_id"] == a
    assert db.search_fts("beta", "alice") == []
    assert db.get_document(a, "bob") is None
    assert db.delete_document(a, "bob") is False


def test_fts_query_is_sanitised(db: SqliteDatabase) -> None:
    doc = _doc(db)
    db.replace_chunks(doc, "local", [{"text": "npm ERR! ERESOLVE"}])
    assert fts_match_query('foo" OR NOT (') == '"foo" OR "OR" OR "NOT"'
    assert db.search_fts('"( * AND', "local") == []  # must not raise
    assert len(db.search_fts("ERESOLVE", "local")) == 1


def test_job_lifecycle(db: SqliteDatabase) -> None:
    doc = _doc(db)
    job = db.create_job(doc)
    assert db.get_job(job)["status"] == "queued"
    db.update_job(job, status="running", stage="extracting", progress=40)
    db.update_job(job, status="failed", error="boom")
    got = db.get_job(job)
    assert (got["status"], got["stage"], got["progress"], got["error"]) == (
        "failed", "extracting", 40, "boom")


def test_paths_with_spaces_and_non_ascii(tmp_path: Path) -> None:
    s = Settings(_env_file=None, kb_data_dir=tmp_path / "my data" / "дані 日本語")
    s.ensure_dirs()
    d = SqliteDatabase(s.db_path)
    d.init_schema()
    doc = _doc(d)
    d.replace_chunks(doc, "local", [{"text": "unicode ok: café"}])
    assert len(d.search_fts("café", "local")) == 1
    assert s.index_dir.is_dir() and s.markdown_dir.is_dir()


def test_config_paths_derive_from_single_data_dir(tmp_path: Path) -> None:
    s = Settings(_env_file=None, kb_data_dir=tmp_path / "d")
    assert s.originals_dir == (tmp_path / "d" / "originals").resolve()
    assert s.frames_dir == (tmp_path / "d" / "frames").resolve()
    assert s.db_path.parent == s.data_dir


def test_config_overrides_and_empty_env_values(tmp_path: Path) -> None:
    s = Settings(_env_file=None, kb_data_dir=tmp_path / "d", kb_frames_dir=tmp_path / "big")
    assert s.frames_dir == (tmp_path / "big").resolve()
    empty = Settings(_env_file=None, kb_data_dir=tmp_path / "d", kb_originals_dir="")
    assert empty.originals_dir == (tmp_path / "d" / "originals").resolve()


def test_relative_data_dir_anchored_to_project_root() -> None:
    assert Settings(_env_file=None, kb_data_dir="./data").data_dir == (PROJECT_ROOT / "data")
