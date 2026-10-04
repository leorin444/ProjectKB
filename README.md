# Local Multimodal Knowledge Base

Media -> Markdown -> RAG, fully local, private, and running completely offline on Windows.

## Key Features

- **Multimodal Ingestion**:
  - **Documents**: PDFs, Markdown, Plain Text, CSV.
  - **Images**: PNG, JPG, WEBP, BMP analyzed via Ollama Vision Language Models (`qwen2.5vl:7b`).
  - **Audio**: MP3, WAV, M4A, FLAC transcribed with precise segment timestamps via `faster-whisper`.
  - **Video**: MP4, MKV, WebM with ffmpeg scene detection, perceptual dHash deduplication, visual keyframe analysis, and unified chronological dialogue + visual merging.
- **Asynchronous Processing Queue**:
  - Single-worker `ThreadPoolExecutor` protects CPU/VRAM from concurrent heavy Whisper/VLM loads.
  - Persistent SQLite job tracking (`queued`, `processing`, `completed`, `failed`, `cancelled`) with cooperative job cancellation and rerun capabilities.
- **Hybrid Retrieval & Grounded Q&A**:
  - Qdrant embedded vector search + SQLite FTS5 BM25 keyword search fused via Reciprocal Rank Fusion (RRF).
  - Optional Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - Strict grounded answer generation citing `[Document Title - Section (HH:MM:SS)]`.
  - Score gating: unanswerable distractors below `min_relevance` automatically return `{"found": false}` without hallucinating.
- **Streamlit Web UI**:
  - Clean 3-tab interface for uploading files, browsing/editing/deleting documents, and asking grounded questions.

---

## Quickstart Setup

### Prerequisites
- Windows 10/11 64-bit
- PowerShell 5.1+
- NVIDIA GPU recommended (8GB+ VRAM for local Ollama models)

### Automated Setup
```powershell
cd A:\Projects\ProjectKB
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -ProjectPath "A:\Projects\ProjectKB"
.\.venv\Scripts\Activate.ps1
```

Pull the required Ollama models:
```powershell
ollama pull llama3.1:8b
ollama pull qwen2.5vl:7b
```

---

## Running the Application

### 1. Start the API Server
```powershell
.\.venv\Scripts\python.exe scripts/run_api.py
```
Or directly with uvicorn:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Health Check: http://127.0.0.1:8000/health
- Interactive OpenAPI Docs: http://127.0.0.1:8000/docs

> **Note**: Qdrant embedded mode locks `data/index` to one process. Only run one API instance at a time.

### 2. Start the Streamlit UI
In a separate terminal:
```powershell
.\.venv\Scripts\python.exe scripts/run_ui.py
```
Open http://localhost:8501 in your browser.

---

## Evaluation Benchmark

Run the automated retrieval and rejection benchmark:
```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --host http://127.0.0.1:8000 --api-key change-me
```
Reports Hit@1, Hit@3, Hit@5, MRR, and Distractor Rejection Rate.

---

## Testing & Code Quality

```powershell
# Run all unit and integration tests (fast test doubles)
.\.venv\Scripts\python.exe -m pytest

# Run linter
.\.venv\Scripts\python.exe -m ruff check .
```

---

## Configuration & Architecture

All configuration is managed through environment variables or `.env`.
Key settings:
- `KB_DATA_DIR`: Base storage directory for SQLite database, Qdrant index, originals, and frames.
- `LLM_MODEL`: Ollama LLM model name (default: `llama3.1:8b`).
- `VISION_MODEL`: Ollama vision model name (default: `qwen2.5vl:7b`).
- `WHISPER_MODEL`: faster-whisper model size (default: `small`).
- `USE_RERANKER`: Enable local cross-encoder reranker (default: `false`).
- `MIN_RELEVANCE`: Strict relevance threshold for distractor rejection (default: `0.25`).

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for architectural decisions and [`docs/API.md`](docs/API.md) for endpoint documentation.
