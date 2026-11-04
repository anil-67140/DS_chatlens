"""Experiment tracking. MLflow (file store, so it lives on Drive and needs no server) when installed,
otherwise an equivalent JSONL log. Records parameters, metrics, timings, dataset fingerprint, code
version and artifact paths for every benchmark run."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .checkpoint import atomic_write_text


def dataset_fingerprint(df: pd.DataFrame, cols=("text", "label")) -> str:
    """Order-independent content hash of a dataframe (dataset version id)."""
    h = hashlib.sha256()
    for row in sorted(map(tuple, df[list(cols)].astype(str).itertuples(index=False, name=None))):
        h.update("\x1f".join(row).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "chatlens": __version__}
    for pkg in ("pandas", "numpy", "scikit-learn", "scipy", "torch", "transformers", "mlflow"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:        # noqa: BLE001
            pass
    return info


class Tracker:
    """MLflow (SQLite backend; the file store is deprecated in MLflow 3) with a JSONL fallback.

    SQLite does not behave reliably on network filesystems such as Google Drive, so pass
    ``local_dir`` (e.g. ``/content/mlflow``) for the live database and call :meth:`sync` to copy it
    atomically into ``root`` (Drive). Artifacts are stored under ``root/mlartifacts``."""

    def __init__(self, root: str | Path, experiment: str = "chatlens", use_mlflow: bool = True,
                 local_dir: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.local = Path(local_dir) if local_dir else self.root
        self.local.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self.mlflow = None
        self.db = self.local / "mlflow.db"
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(f"sqlite:///{self.db}")
                if mlflow.get_experiment_by_name(experiment) is None:
                    mlflow.create_experiment(experiment, artifact_location=str(self.root / "mlartifacts"))
                mlflow.set_experiment(experiment)
                self.mlflow = mlflow
            except Exception:        # noqa: BLE001 - tracking must never break an experiment
                self.mlflow = None
        self.jsonl = self.root / "runs.jsonl"

    def sync(self) -> Path | None:
        """Copy the live MLflow database into ``root`` (atomic). No-op if they are the same place."""
        if self.mlflow and self.db.exists() and self.local.resolve() != self.root.resolve():
            from .checkpoint import atomic_write_bytes
            return atomic_write_bytes(self.root / "mlflow.db", self.db.read_bytes())
        return None

    @property
    def backend(self) -> str:
        return "mlflow" if self.mlflow else "jsonl"

    def log_run(self, name: str, params: dict, metrics: dict, artifacts: list[str | Path] = (),
                tags: dict | None = None) -> None:
        params = {k: (v if isinstance(v, (int, float, str, bool)) else json.dumps(v, default=str))
                  for k, v in params.items()}
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        rec = {"name": name, "experiment": self.experiment, "params": params, "metrics": clean,
               "artifacts": [str(a) for a in artifacts], "tags": tags or {}, "env": environment_info()}
        prev = self.jsonl.read_text(encoding="utf-8") if self.jsonl.exists() else ""
        atomic_write_text(self.jsonl, prev + json.dumps(rec, default=str) + "\n")     # always keep a plain log
        if self.mlflow:
            with self.mlflow.start_run(run_name=name):
                self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self.mlflow.log_metrics(clean)
                self.mlflow.set_tags({**(tags or {}), "chatlens": __version__})
                for a in artifacts:
                    if Path(a).exists():
                        self.mlflow.log_artifact(str(a))

    def log_benchmark(self, task: str, metrics: dict[str, dict], data_fingerprint: str, seed: int,
                      store_root: str | Path) -> int:
        n = 0
        for name, m in metrics.items():
            flat = {k: m.get(k) for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc",
                                          "roc_auc_macro_ovr", "pr_auc_macro", "ece", "brier", "log_loss",
                                          "fit_seconds", "inference_seconds")}
            art = Path(store_root) / "nlp" / task / "".join(c if c.isalnum() or c in "-_+" else "_" for c in name)
            self.log_run(f"{task}/{name}",
                         {"task": task, "method": name, "family": m["family"], "zero_shot": m["zero_shot"],
                          "seed": seed, "dataset": data_fingerprint, **{f"hp_{k}": v for k, v in
                                                                         (m.get("best_params") or {}).items()}},
                         flat, [art / "result.json"], {"task": task})
            n += 1
        return n
