"""Crash-safe checkpointing for Colab/Drive.

Guarantees
----------
* **Atomic writes**: data is written to ``<name>.tmp-<pid>`` in the same directory, fsync'd and
  moved with ``os.replace``. A disconnect mid-write leaves either the old file or no file,
  never a half-written one.
* **Manifest**: ``manifest.json`` (itself written atomically) records every stage with status,
  parameters hash, timings and a SHA-256 for every artifact.
* **Verified resume**: a stage is skipped only if the manifest says ``complete``, the parameters
  hash is unchanged, and every artifact still exists with the recorded checksum. Anything else
  (missing file, edited file, changed parameters, stage left in ``running`` by a crash) reruns
  the stage and says why.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_utils import get_logger

log = get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def params_hash(params: dict | None) -> str:
    blob = json.dumps(params or {}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str))


def cleanup_tmp(root: str | Path) -> int:
    """Remove orphaned temp files left by an interrupted write."""
    n = 0
    for p in Path(root).rglob("*.tmp-*"):
        if p.is_file():
            p.unlink(missing_ok=True)
            n += 1
    return n


@dataclass
class StageResult:
    name: str
    skipped: bool
    reason: str
    seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)


class CheckpointStore:
    """A directory (e.g. on Google Drive) holding artifacts plus a manifest of completed stages."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        removed = cleanup_tmp(self.root)
        if removed:
            log.info("removed %d orphaned temp file(s)", removed)
        self._manifest = self._load()

    # -- manifest ---------------------------------------------------------------------------
    def _load(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A corrupt manifest must not destroy the work: keep it aside and start clean.
                self.manifest_path.rename(self.manifest_path.with_suffix(".corrupt.json"))
                log.warning("manifest was corrupt; moved aside, stages will be re-verified")
        return {"version": 1, "created": _now(), "stages": {}}

    def _save(self) -> None:
        self._manifest["updated"] = _now()
        atomic_write_json(self.manifest_path, self._manifest)

    @property
    def stages(self) -> dict:
        return self._manifest["stages"]

    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"artifact path escapes the store: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- verification -----------------------------------------------------------------------
    def verify(self, name: str, params: dict | None = None) -> tuple[bool, str]:
        st = self.stages.get(name)
        if not st:
            return False, "never run"
        if st["status"] != "complete":
            return False, f"previous run did not finish (status={st['status']})"
        if params is not None and st.get("params_hash") != params_hash(params):
            return False, "parameters changed"
        for rel, meta in st.get("artifacts", {}).items():
            p = self.root / rel
            if not p.exists():
                return False, f"artifact missing: {rel}"
            if p.stat().st_size != meta["bytes"] or sha256_file(p) != meta["sha256"]:
                return False, f"artifact failed checksum: {rel}"
        return True, "verified"

    # -- running ----------------------------------------------------------------------------
    def run(self, name: str, fn: Callable[[], Any], *, outputs: Iterable[str] | Callable[[], Iterable[str]],
            params: dict | None = None, force: bool = False,
            depends_on: Iterable[str] = ()) -> StageResult:
        """Run ``fn`` unless a verified checkpoint exists. ``fn`` must write every path in
        ``outputs`` (relative to the store root), preferably with the atomic_* helpers.
        ``outputs`` may be a callable, evaluated AFTER ``fn`` ran, for artifacts whose names are only
        known once the work is done (e.g. fitted-model files)."""
        outputs_spec = outputs
        for dep in depends_on:
            ok, why = self.verify(dep)
            if not ok:
                raise RuntimeError(f"stage '{name}' depends on '{dep}' which is not valid: {why}")
        if not force:
            ok, why = self.verify(name, params)
            if ok:
                log.info("stage=%s SKIPPED (verified checkpoint)", name)
                return StageResult(name, True, why, artifacts=list(self.stages[name]["artifacts"]))
            log.info("stage=%s RUNNING (%s)", name, why)
        self.stages[name] = {"status": "running", "started": _now(),
                             "params_hash": params_hash(params), "params": params or {},
                             "artifacts": {}}
        self._save()
        t0 = time.perf_counter()
        try:
            fn()
            outputs = list(outputs_spec() if callable(outputs_spec) else outputs_spec)
            arts = {}
            for rel in outputs:
                p = self.root / rel
                if not p.exists():
                    raise FileNotFoundError(f"stage '{name}' did not produce declared output {rel}")
                arts[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        except BaseException as exc:      # includes KeyboardInterrupt from a Colab "stop"
            self.stages[name].update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                                     finished=_now())
            self._save()
            raise
        secs = round(time.perf_counter() - t0, 3)
        self.stages[name].update(status="complete", finished=_now(), seconds=secs, artifacts=arts)
        self._save()
        log.info("stage=%s COMPLETE seconds=%s artifacts=%d", name, secs, len(arts))
        return StageResult(name, False, "ran", secs, list(arts))

    # -- convenience IO (all atomic) --------------------------------------------------------
    def save_json(self, rel: str, obj: Any) -> Path:
        return atomic_write_json(self.path(rel), obj)

    def load_json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def save_df(self, rel: str, df) -> Path:
        import io

        import joblib
        buf = io.BytesIO()
        joblib.dump(df, buf, compress=3)
        return atomic_write_bytes(self.path(rel), buf.getvalue())

    def load_df(self, rel: str):
        import joblib
        return joblib.load(self.path(rel))

    def save_csv(self, rel: str, df, **kw) -> Path:
        return atomic_write_text(self.path(rel), df.to_csv(index=kw.pop("index", False), **kw))

    def save_object(self, rel: str, obj: Any) -> Path:
        return self.save_df(rel, obj)

    def load_object(self, rel: str) -> Any:
        return self.load_df(rel)

    def save_figure(self, rel: str, fig, **kw) -> Path:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format=Path(rel).suffix.lstrip(".") or "png", bbox_inches="tight",
                    dpi=kw.pop("dpi", 150), **kw)
        return atomic_write_bytes(self.path(rel), buf.getvalue())

    def summary(self) -> list[dict]:
        return [{"stage": k, "status": v["status"], "seconds": v.get("seconds"),
                 "artifacts": len(v.get("artifacts", {})), "finished": v.get("finished")}
                for k, v in self.stages.items()]
