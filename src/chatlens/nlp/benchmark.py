"""Resumable benchmark runner: one checkpointed stage per (task, method).

Each method's test predictions, probabilities, metrics, timings and (for sklearn models) the fitted
model are saved as soon as that method finishes, so a Colab disconnect loses at most the method
that was running. Selection of hyper-parameters uses the validation split; the test split is only
used for the final numbers."""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..checkpoint import CheckpointStore
from ..logging_utils import get_logger
from . import evaluation as ev
from .classifiers import BaseClassifier, HFFineTunedClassifier, SklearnTextClassifier

log = get_logger()


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_+" else "_" for c in name)


def run_benchmark(task: str, splits: dict[str, pd.DataFrame], methods: Sequence[BaseClassifier],
                  store: CheckpointStore, seed: int = 42, n_boot: int = 1000,
                  skip_on_error: bool = True, data_fingerprint: str = "") -> dict:
    """Run/resume every method. Returns {'metrics': {...}, 'errors': {...}, 'classes': [...]}."""
    classes = sorted(splits["train"]["label"].unique())
    errors: dict[str, str] = {}
    for m in methods:
        base = f"nlp/{task}/{_safe(m.name)}"
        outs = [f"{base}/result.json", f"{base}/test_pred.pkl"]
        extra: list[str] = []          # model files: known only after fitting, but must be checksummed too
        params = {"task": task, "method": m.name, "seed": seed, "n_train": len(splits["train"]),
                  "n_test": len(splits["test"]), "data": data_fingerprint,
                  "classes": classes, "code_version": __version__}

        def work(m=m, base=base, extra=extra):
            t0 = time.perf_counter()
            m.fit(splits["train"], splits["val"])
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            texts = splits["test"]["text"].tolist()
            proba = m.predict_proba(texts) if m.has_proba else None
            pred = (np.array(m.classes_)[proba.argmax(1)] if proba is not None
                    else m.predict(texts))
            infer_s = time.perf_counter() - t1
            met = ev.classification_metrics(splits["test"]["label"], pred, classes, proba)
            est, lo, hi = ev.bootstrap_metric_ci(splits["test"]["label"], pred, classes,
                                                 "macro_f1", n_boot, seed=seed)
            met.update({"macro_f1_ci95": [lo, hi], "method": m.name, "family": m.family,
                        "zero_shot": m.zero_shot, "fit_seconds": round(fit_s, 3),
                        "inference_seconds": round(infer_s, 3),
                        "ms_per_1k_texts": round(1000 * infer_s / max(len(texts), 1) * 1000, 2),
                        "best_params": m.best_params_,
                        "val_macro_f1": getattr(m, "val_macro_f1", None)})
            store.save_json(f"{base}/result.json", met)
            store.save_object(f"{base}/test_pred.pkl", {"pred": pred, "proba": proba,
                                                        "classes": list(m.classes_)})
            if isinstance(m, SklearnTextClassifier):
                store.save_object(f"{base}/model.joblib", m.model)
                extra.append(f"{base}/model.joblib")
            elif isinstance(m, HFFineTunedClassifier):
                w = Path(m.ckpt_dir) / "best.pt"
                w = w if w.exists() else Path(m.ckpt_dir) / "last.pt"
                try:
                    extra.append(str(w.resolve().relative_to(store.root.resolve())))
                except ValueError:      # checkpoint dir outside the store: cannot be tracked by the manifest
                    pass

        try:
            store.run(f"{task}:{m.name}", work, outputs=lambda outs=outs, extra=extra: outs + extra, params=params)
        except Exception as exc:        # noqa: BLE001 - a single failing model must not stop the run
            errors[m.name] = f"{type(exc).__name__}: {str(exc)[:300]}"
            log.warning("method %s failed: %s", m.name, errors[m.name])
            if not skip_on_error:
                raise
    metrics = {}
    for m in methods:
        p = store.root / f"nlp/{task}/{_safe(m.name)}/result.json"
        if p.exists() and m.name not in errors:
            metrics[m.name] = store.load_json(f"nlp/{task}/{_safe(m.name)}/result.json")
    return {"metrics": metrics, "errors": errors, "classes": classes}


def comparison_table(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, m in metrics.items():
        rows.append({
            "method": name, "family": m["family"], "zero_shot": m["zero_shot"],
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "macro_f1_ci_low": m["macro_f1_ci95"][0], "macro_f1_ci_high": m["macro_f1_ci95"][1],
            "weighted_f1": m["weighted_f1"], "balanced_acc": m["balanced_accuracy"],
            "mcc": m["mcc"], "roc_auc_ovr": m.get("roc_auc_macro_ovr"),
            "pr_auc": m.get("pr_auc_macro"), "ece": m.get("ece"), "brier": m.get("brier"),
            "fit_s": m["fit_seconds"], "ms_per_1k": m["ms_per_1k_texts"]})
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def significance_table(task: str, y_true, store: CheckpointStore, classes, methods: Sequence[str],
                       n_boot: int = 1000, seed: int = 42, reference: str | None = None):
    preds = {}
    for name in methods:
        p = store.root / f"nlp/{task}/{_safe(name)}/test_pred.pkl"
        if p.exists():
            preds[name] = store.load_object(f"nlp/{task}/{_safe(name)}/test_pred.pkl")["pred"]
    rows = ev.compare_all(y_true, preds, classes, reference, n_boot, seed)
    return pd.DataFrame(rows)
