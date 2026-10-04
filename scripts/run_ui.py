"""Run the Streamlit UI frontend.

Usage:
    python scripts/run_ui.py [--port 8501]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "ui" / "streamlit_app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Knowledge Base Streamlit UI")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.port",
        str(args.port),
    ]

    print(f"Starting Knowledge Base UI on port {args.port}...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping UI...")


if __name__ == "__main__":
    main()
