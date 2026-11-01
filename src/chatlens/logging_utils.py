"""Structured logging. Message content must never be passed to a logger."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


def get_logger(name: str = "chatlens") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(logger: logging.Logger, stage: str) -> Iterator[dict]:
    """Log stage duration and row counts only (never content)."""
    info: dict = {}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = round(time.perf_counter() - t0, 4)
        logger.info("stage=%s seconds=%s %s", stage, info["seconds"],
                    {k: v for k, v in info.items() if k != "seconds"})
