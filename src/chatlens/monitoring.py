"""Monitoring: processing time/resources, NLP output quality, drift (only for trained models),
and a privacy-safe feedback log.

Drift monitoring is meaningful ONLY for components that are trained models (the sentiment/emotion
classifiers). Rule-based parts (parser, frequency analytics) have no model to drift; they are
covered by data-quality validation instead. Drift here means: does the distribution of a model's
predictions / confidence on a NEW chat differ from its distribution on the held-out reference
data? That flags "this chat is unlike what the model was evaluated on", which is a reason to
distrust its output. It does not prove the model is wrong."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:           # pragma: no cover
    psutil = None


@contextmanager
def resource_monitor(log: dict, name: str):
    """Record wall time, CPU time and RSS memory delta of a block into ``log[name]``."""
    proc = psutil.Process() if psutil else None
    rss0 = proc.memory_info().rss if proc else 0
    c0, t0 = time.process_time(), time.perf_counter()
    try:
        yield
    finally:
        log[name] = {"wall_s": round(time.perf_counter() - t0, 4), "cpu_s": round(time.process_time() - c0, 4),
                     "rss_mb": round(proc.memory_info().rss / 1e6, 1) if proc else None,
                     "rss_delta_mb": round((proc.memory_info().rss - rss0) / 1e6, 1) if proc else None}


def psi(reference: dict[str, float], current: dict[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index between two categorical distributions.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    keys = sorted(set(reference) | set(current))
    r = np.array([reference.get(k, 0.0) for k in keys], float)
    c = np.array([current.get(k, 0.0) for k in keys], float)
    r, c = np.clip(r / max(r.sum(), eps), eps, None), np.clip(c / max(c.sum(), eps), eps, None)
    return float(((c - r) * np.log(c / r)).sum())


def nlp_output_checks(pred_labels: list[str], confidences: np.ndarray | None, reference_dist: dict[str, float],
                      reference_confidence: float | None = None, n_total: int | None = None) -> dict:
    """Quality checks on a model's outputs for one chat."""
    n = len(pred_labels)
    cnt = Counter(pred_labels)
    dist = {k: v / n for k, v in cnt.items()} if n else {}
    res: dict = {"n_scored": n, "coverage": (n / n_total) if n_total else None, "predicted_distribution": dist}
    alerts = []
    if n < 30:
        alerts.append("fewer than 30 scored messages: distribution comparisons are unreliable")
    if n:
        res["psi_vs_reference"] = psi(reference_dist, dist)
        if res["psi_vs_reference"] > 0.25:
            alerts.append("major shift in predicted-label distribution vs reference (PSI > 0.25)")
        elif res["psi_vs_reference"] > 0.1:
            alerts.append("moderate shift in predicted-label distribution (PSI 0.1-0.25)")
        if max(dist.values()) > 0.9 and n >= 30:
            alerts.append("a single class makes up > 90% of predictions (possible degenerate output)")
    if confidences is not None and len(confidences):
        res["mean_confidence"] = float(np.mean(confidences))
        res["low_confidence_share"] = float(np.mean(np.asarray(confidences) < 0.5))
        if reference_confidence is not None and res["mean_confidence"] < reference_confidence - 0.15:
            alerts.append("mean confidence is much lower than on the reference data")
    res["alerts"] = alerts
    return res


class FeedbackStore:
    """Append-only JSONL feedback log. Stores a rating, a section name and a short free-text note
    that the user typed. It NEVER receives chat content from the app: callers pass only these
    fields, and the note is truncated and checked against a deny-list of message text if given."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, section: str, rating: int, comment: str = "", forbidden_substrings: list[str] | None = None) -> dict:
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError("rating must be 1-5")
        comment = comment.strip()[:500]
        for bad in forbidden_substrings or []:
            if bad and len(bad) > 8 and bad.lower() in comment.lower():
                comment = "[removed: looked like pasted chat content]"
                break
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "section": section[:50],
               "rating": rating, "comment": comment}
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def summary(self) -> dict:
        recs = self.read()
        by: dict[str, list[int]] = {}
        for r in recs:
            by.setdefault(r["section"], []).append(r["rating"])
        return {s: {"n": len(v), "mean_rating": float(np.mean(v))} for s, v in by.items()}
