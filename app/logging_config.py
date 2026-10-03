"""Structured key=value logging. Never pass document text or secrets to the logger."""

from __future__ import annotations

import logging
import sys


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"ts={self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} level={record.levelname} "
            f"logger={record.name} msg={record.getMessage()!r}"
        )
        extras = getattr(record, "kv", None)
        if extras:
            base += " " + " ".join(f"{k}={v!r}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def log_kv(logger: logging.Logger, level: int, msg: str, **kv: object) -> None:
    logger.log(level, msg, extra={"kv": kv})
