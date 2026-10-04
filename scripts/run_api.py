"""Run the FastAPI backend server.

Usage:
    python scripts/run_api.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Start the Knowledge Base API server")
    parser.add_argument("--host", default=settings.api_host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for development")
    args = parser.parse_args()

    print(f"Starting Knowledge Base API at http://{args.host}:{args.port}")
    print("Documentation available at /docs and health check at /health")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
