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
| 11 | Single-worker ThreadPoolExecutor for background jobs | Protects CPU and RAM from concurrent heavy Whisper/VLM loads while maintaining an isolated queue with SQLite state tracking. |
| 12 | Video scene detection via ffmpeg `gt(scene, 0.3)` + periodic fallback | Efficient keyframe extraction without running VLM on every raw video frame. |
| 13 | Perceptual dHash (64-bit difference hash) with Hamming distance threshold <= 6 | Discards redundant, near-identical keyframes before expensive VLM inference. |
| 14 | Multimodal time-merging of audio transcript + visual keyframe descriptions | Integrates spoken dialogue and on-screen visuals (slides, code errors, diagrams) into coherent chronological sections with grounded timestamps. |
| 15 | Cooperative job cancellation and rerun | Allows long-running indexing tasks to be gracefully aborted or restarted without leaving inconsistent database state. |
