# API

Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`.
All non-health endpoints (added in later phases) require the `X-API-Key` header.

## Implemented Endpoints

### System
- `GET /health` (unauthenticated) - inspect readiness of Ollama, DB, Vector store, ffmpeg.

### Ingestion & Jobs
- `POST /ingest/file` - Upload file (multipart). Sync HTTP 200 for docs/images; async HTTP 202 with `job_id` for audio/video or `background=true`.
- `GET /jobs/{job_id}` - Poll status (`queued`, `processing`, `completed`, `failed`, `cancelled`), progress (0-100), and stage details.
- `POST /jobs/{job_id}/cancel` - Request cooperative cancellation of an active or queued job.
- `POST /jobs/{job_id}/rerun` - Resubmit a job for processing.

### Documents & Re-indexing
- `GET /documents` - List user documents with optional `type`, `tag`, and pagination filters.
- `GET /documents/{doc_id}` - Retrieve metadata and markdown transcript/content.
- `DELETE /documents/{doc_id}` - Delete original files, extracted keyframes, SQLite chunks, and Qdrant points.
- `PUT /documents/{doc_id}/markdown` - Update transcribed markdown and automatically re-index chunks.
- `POST /documents/{doc_id}/reindex` - Re-chunk, re-embed, and re-index an existing document.

### Retrieval & Grounded Query
- `POST /search` - Hybrid vector + FTS5 retrieval with RRF ranking (inspect chunks and relevance scores).
- `POST /query` - Strict grounded question-answering with citations (`[Doc Title - Section (HH:MM:SS)]`).
