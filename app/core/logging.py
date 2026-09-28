"""Logging setup shared by the API, the CLI and the scrapers."""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings

_CONFIGURED = False


def setup_logging(level: int | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    resolved = level if level is not None else (logging.DEBUG if settings.debug else logging.INFO)

    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(resolved)
    root.handlers = [handler]

    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
