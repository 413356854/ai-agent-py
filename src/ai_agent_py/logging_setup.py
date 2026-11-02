"""Logging bootstrap (stdlib logging, readable one-line format)."""

from __future__ import annotations

import logging
import sys

from ai_agent_py.config import get_settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once; `level` overrides `LOG_LEVEL`."""
    resolved = (level or get_settings().log_level).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))
    # These libraries are chatty at INFO level.
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
