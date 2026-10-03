# API

Interactive OpenAPI docs: `http://127.0.0.1:8000/docs`.
All non-health endpoints (added in later phases) require the `X-API-Key` header.

## Implemented (Phase 0)

### `GET /health`
Unauthenticated. Always returns HTTP 200; inspect the body.

```json
{
  "status": "ok | degraded",
  "components": {
    "ollama":       {"ok": true,  "detail": "reachable, 2 model(s)"},
    "llm_model":    {"ok": true,  "detail": "llama3.1:8b available"},
    "vision_model": {"ok": false, "detail": "model qwen2.5vl:7b not found - run `ollama pull qwen2.5vl:7b`"},
    "database":     {"ok": true,  "detail": "sqlite ok"},
    "vector_store": {"ok": true,  "detail": "qdrant ok"},
    "ffmpeg":       {"ok": true,  "detail": "ffmpeg version ..."}
  }
}
```

## Planned
Ingest, jobs, documents, query/search endpoints: see the project brief, Section 8.
