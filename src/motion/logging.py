"""Small logging helper so every module logs consistently (and prettily via rich
when available, plain otherwise)."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "motion") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.getenv("MOTION_LOG_LEVEL", "INFO").upper()
        try:
            from rich.logging import RichHandler  # noqa: PLC0415

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
        logging.basicConfig(level=level, format=fmt, handlers=[handler], datefmt="%H:%M:%S")
        _CONFIGURED = True
    return logging.getLogger(name)
