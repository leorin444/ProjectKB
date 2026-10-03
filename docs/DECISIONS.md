# Design decisions

| # | Decision | Why |
|---|---|---|
| 1 | stdlib `sqlite3`, one short-lived connection per unit of work, all SQL in `storage/db.py` | No ORM dependency; thread-safe for the in-process worker; releases file handles promptly on Windows. Another backend can replace the class later. |
| 2 | Chunk timestamps stored as REAL seconds | Easy arithmetic and sorting; formatted to `HH:MM:SS` only in Markdown/API. |
| 3 | FTS5 as an external-content table kept in sync by triggers | No duplicate text storage; cascade deletes also clean the index. |
| 4 | `replace_chunks` deletes then inserts in one transaction | Idempotent re-index; no duplicates. |
| 5 | Relative `KB_DATA_DIR` is anchored at the project root, not the cwd | Same data folder regardless of where uvicorn/pytest is started. |
| 6 | Qdrant payload indexes skipped in embedded mode | They are a no-op locally; add them in the server implementation. |
| 7 | `LLMClient` methods other than health raise `NotImplementedError` in Phase 0 | They are implemented and verified end-to-end in Phase 1. |
| 8 | `/health` is unauthenticated and never raises; each dependency is reported separately | Per brief; `status` is `ok` only if all components are ok, else `degraded`. |
| 9 | Auth is one function, `get_principal`, returning `Principal(owner_id)` | Routes never read the API key, so real auth can replace it without touching routes. |
| 10 | Ruff `T20` enabled | Enforces "no print in app code". |
