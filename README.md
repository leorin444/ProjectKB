# Local Multimodal Knowledge Base

Media -> Markdown -> RAG, fully local. Windows-native, PowerShell. Currently: **Phase 0 (scaffold)**.

## Setup

```powershell
cd A:\Projects\ProjectKB
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -ProjectPath "A:\Projects\ProjectKB"
.\.venv\Scripts\Activate.ps1
```

`setup.ps1` installs Python 3.11, ffmpeg and Ollama, builds `.venv`, and creates `.env`.
Pull models if you skipped them: `ollama pull llama3.1:8b; ollama pull qwen2.5vl:7b`.

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/health or http://127.0.0.1:8000/docs.

> Qdrant embedded mode locks `data\index` to one process. Run only one API instance.

## UI (from Phase 1)

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\streamlit_app.py
```

## Tests and lint

```powershell
.\.venv\Scripts\python.exe -m pytest              # all tests
.\.venv\Scripts\python.exe -m pytest -m "not slow" # skip model-dependent tests
.\.venv\Scripts\python.exe -m ruff check .
```

## Configuration

Copy `.env.example` to `.env`. All paths derive from `KB_DATA_DIR`; `KB_ORIGINALS_DIR` and
`KB_FRAMES_DIR` can point to another drive. See `docs/DECISIONS.md` and `docs/API.md`.
