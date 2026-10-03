"""SQLite metadata + FTS5 storage (stdlib sqlite3).

All SQL lives here so another backend (e.g. Postgres) can replace this class later.
Every chunk query takes an `owner_id` - there is deliberately no unscoped search.
Timestamps (`start_ts`/`end_ts`) are stored as REAL seconds; formatting to HH:MM:SS
happens at the Markdown / API edge.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    title         TEXT NOT NULL,
    original_name TEXT,
    source_path   TEXT,
    md_path       TEXT,
    tags          TEXT NOT NULL DEFAULT '[]',   -- JSON array
    owner_id      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id, type);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    owner_id    TEXT NOT NULL,
    section     TEXT,
    start_ts    REAL,
    end_ts      REAL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_owner ON chunks(owner_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT NOT NULL DEFAULT 'queued',
    progress    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def fts_match_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query (no operator injection)."""
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens)


class SqliteDatabase:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        # One short-lived connection per unit of work: thread-safe and releases the
        # file handle promptly (matters on Windows).
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- lifecycle -------------------------------------------------------
    def init_schema(self) -> None:
        """Idempotent: safe to call on every startup."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 0:
                conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))

    def ping(self) -> None:
        with self.connect() as conn:
            conn.execute("SELECT 1 FROM documents LIMIT 1")

    # --- documents -------------------------------------------------------
    def insert_document(
        self, *, owner_id: str, type: str, title: str, original_name: str | None = None,
        source_path: str | None = None, md_path: str | None = None, tags: str = "[]",
        status: str = "pending", id: str | None = None,
    ) -> str:
        doc_id, now = id or new_id(), utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO documents(id,type,title,original_name,source_path,md_path,tags,"
                "owner_id,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, type, title, original_name, source_path, md_path, tags, owner_id,
                 status, now, now),
            )
        return doc_id

    def get_document(self, doc_id: str, owner_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id=? AND owner_id=?", (doc_id, owner_id)
            ).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str, owner_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE id=? AND owner_id=?", (doc_id, owner_id)
            )
        return cur.rowcount > 0

    # --- chunks ----------------------------------------------------------
    def replace_chunks(
        self, document_id: str, owner_id: str, chunks: Sequence[dict[str, Any]]
    ) -> None:
        """Atomically swap a document's chunks (idempotent re-index, no duplicates)."""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE document_id=? AND owner_id=?", (document_id, owner_id)
            )
            conn.executemany(
                "INSERT INTO chunks(id,document_id,owner_id,section,start_ts,end_ts,text,"
                "token_count) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (c.get("id") or new_id(), document_id, owner_id, c.get("section"),
                     c.get("start_ts"), c.get("end_ts"), c["text"], c.get("token_count", 0))
                    for c in chunks
                ],
            )

    def count_chunks(self, document_id: str | None = None) -> int:
        with self.connect() as conn:
            if document_id:
                q, a = "SELECT COUNT(*) FROM chunks WHERE document_id=?", (document_id,)
            else:
                q, a = "SELECT COUNT(*) FROM chunks", ()
            return int(conn.execute(q, a).fetchone()[0])

    def search_fts(self, query: str, owner_id: str, limit: int = 20) -> list[dict[str, Any]]:
        match = fts_match_query(query)
        if not match:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.id, c.document_id, c.section, c.start_ts, c.end_ts, c.text, "
                "bm25(chunks_fts) AS score FROM chunks_fts "
                "JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? AND c.owner_id = ? ORDER BY score LIMIT ?",
                (match, owner_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- jobs ------------------------------------------------------------
    def create_job(self, document_id: str | None) -> str:
        job_id, now = new_id(), utcnow()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(id,document_id,created_at,updated_at) VALUES (?,?,?,?)",
                (job_id, document_id, now, now),
            )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def update_job(
        self, job_id: str, *, status: str | None = None, stage: str | None = None,
        progress: int | None = None, error: str | None = None,
    ) -> None:
        fields = {"status": status, "stage": stage, "progress": progress, "error": error}
        sets = {k: v for k, v in fields.items() if v is not None}
        sets["updated_at"] = utcnow()
        assignments = ", ".join(f"{k}=?" for k in sets)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*sets.values(), job_id))  # noqa: S608 - keys are hard-coded above
