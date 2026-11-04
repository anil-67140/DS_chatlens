"""Evaluation metrics, each applied only where it is meaningful.

* Classification (sentiment, emotion): accuracy, balanced accuracy, precision/recall/F1 per class,
  macro and weighted averages, MCC, Cohen's kappa, confusion matrix, one-vs-rest ROC-AUC and
  PR-AUC and calibration (ECE, Brier, log-loss) **only when the method outputs probabilities**.
* Paired model comparison: McNemar (exact for small discordant counts) + paired bootstrap CI of the
  macro-F1 difference + Holm correction.
* Topic models / clustering: NPMI coherence, topic diversity, silhouette, Davies-Bouldin,
  Calinski-Harabasz. These are unsupervised; no "accuracy" is ever computed for them.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from scipy import stats as ss
from sklearn import metrics as skm


# -------------------------------------------------------------------------- classification
def expected_calibration_error(y_idx: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y_true: Sequence, y_pred: Sequence, classes: Sequence,
                           proba: np.ndarray | None = None) -> dict:
    classes = list(classes)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = skm.precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    out: dict = {
        "n": int(len(yt)),
        "accuracy": float(skm.accuracy_score(yt, yp)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(yt, yp)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(skm.f1_score(yt, yp, labels=classes, average="weighted",
                                          zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(yt, yp)),
        "cohen_kappa": float(skm.cohen_kappa_score(yt, yp)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes,
                             "matrix": skm.confusion_matrix(yt, yp, labels=classes).tolist()},
        "has_probabilities": proba is not None,
    }
    if proba is not None:
        proba = np.asarray(proba, float)
        idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([idx[c] for c in yt])
        present = [i for i in range(len(classes)) if (y_idx == i).any()]
        onehot = np.eye(len(classes))[y_idx]
        roc, pr = {}, {}
        for i in present:
            if 0 < onehot[:, i].sum() < len(yt):
                roc[classes[i]] = float(skm.roc_auc_score(onehot[:, i], proba[:, i]))
                pr[classes[i]] = float(skm.average_precision_score(onehot[:, i], proba[:, i]))
        out["roc_auc_per_class"], out["pr_auc_per_class"] = roc, pr
        out["roc_auc_macro_ovr"] = float(np.mean(list(roc.values()))) if roc else None
        out["pr_auc_macro"] = float(np.mean(list(pr.values()))) if pr else None
        clipped = np.clip(proba, 1e-12, 1)
        clipped = clipped / clipped.sum(1, keepdims=True)
        out["log_loss"] = float(skm.log_loss(y_idx, clipped, labels=list(range(len(classes)))))
        out["brier"] = float(((onehot - proba) ** 2).sum(1).mean())
        out["ece"] = expected_calibration_error(y_idx, proba)
    return out


def bootstrap_metric_ci(y_true, y_pred, classes, metric: str = "macro_f1", iterations: int = 1000,
                        alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a label-based metric (resampling test items)."""
    (yt, yp), fn = _prepare(y_true, [y_pred], classes, metric)
    rng = np.random.default_rng(seed)
    est = fn(yt, yp)
    vals = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, len(yt), len(yt))
        vals[i] = fn(yt[idx], yp[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


def _macro_f1_fast(yt: np.ndarray, yp: np.ndarray, k: int) -> float:
    """Macro-F1 from integer labels via bincount (identical to sklearn, ~100x faster, which matters
    inside bootstrap loops)."""
    tp = np.bincount(yt[yt == yp], minlength=k).astype(float)
    pred = np.bincount(yp, minlength=k).astype(float)
    true = np.bincount(yt, minlength=k).astype(float)
    denom = pred + true
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def _prepare(y_true, preds: list, classes, metric: str):
    """Encode labels as ints once; return (encoded arrays, metric function on int arrays)."""
    idx = {c: i for i, c in enumerate(classes)}
    enc = [np.array([idx[v] for v in np.asarray(a)]) for a in [y_true, *preds]]
    k = len(classes)
    if metric == "macro_f1":
        return enc, lambda a, b: _macro_f1_fast(a, b, k)
    if metric == "accuracy":
        return enc, lambda a, b: float((a == b).mean())
    raise ValueError(metric)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar test on paired correctness. Exact binomial when discordant pairs < 25."""
    yt, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    ca, cb = a == yt, b == yt
    n01 = int((ca & ~cb).sum())       # A right, B wrong
    n10 = int((~ca & cb).sum())       # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"a_only_correct": n01, "b_only_correct": n10, "p_value": 1.0, "test": "none"}
    if n < 25:
        p = float(ss.binomtest(min(n01, n10), n, 0.5).pvalue)
        test = "exact_binomial"
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = float(ss.chi2.sf(stat, 1))
        test = "chi2_continuity"
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p, "test": test,
            "effect_(n01-n10)/n": (n01 - n10) / n}


def paired_bootstrap_diff(y_true, pred_a, pred_b, classes, metric="macro_f1",
                          iterations: int = 1000, alpha: float = 0.05,
                          seed: int = 42) -> dict:
    """CI of metric(A) - metric(B) on the same resampled items (paired)."""
    (yt, a, b), fn = _prepare(y_true, [pred_a, pred_b], classes, metric)
    rng = np.random.default_rng(seed)
    d = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, len(yt), len(yt))
        d[i] = fn(yt[idx], a[idx]) - fn(yt[idx], b[idx])
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(fn(yt, a) - fn(yt, b)), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare_all(y_true, preds: dict[str, np.ndarray], classes, reference: str | None = None,
                iterations: int = 1000, seed: int = 42) -> list[dict]:
    """Each method vs a reference (default: best macro-F1) with Holm-adjusted McNemar."""
    f1 = {}
    for k, v in preds.items():
        (yt_i, yp_i), fn = _prepare(y_true, [v], classes, "macro_f1")
        f1[k] = fn(yt_i, yp_i)
    ref = reference or max(f1, key=f1.get)
    rows = []
    for k in preds:
        if k == ref:
            continue
        mc = mcnemar(y_true, preds[ref], preds[k])
        bd = paired_bootstrap_diff(y_true, preds[ref], preds[k], classes, iterations=iterations,
                                   seed=seed)
        rows.append({"reference": ref, "method": k, "mcnemar_p": mc["p_value"],
                     "macro_f1_diff": bd["diff"], "diff_ci_low": bd["ci_low"],
                     "diff_ci_high": bd["ci_high"]})
    adj = holm([r["mcnemar_p"] for r in rows]) if rows else []
    for r, p in zip(rows, adj, strict=True):
        r["holm_p"] = p
        r["practically_meaningful"] = abs(r["macro_f1_diff"]) >= 0.01
        r["verdict"] = ("indistinguishable" if p >= 0.05 else
                        "reference better" if r["macro_f1_diff"] > 0 else "method better")
    return rows


# ------------------------------------------------------------------- topics and clustering
def topic_diversity(topics: list[list[str]], top_k: int = 10) -> float:
    """Share of unique words among the top-k words of all topics (1 = no overlap)."""
    words = [w for t in topics for w in t[:top_k]]
    return len(set(words)) / len(words) if words else 0.0


def npmi_coherence(topics: list[list[str]], docs_tokens: list[list[str]], top_k: int = 10,
                   eps: float = 1e-12) -> dict:
    """Mean NPMI coherence over topics (document co-occurrence, Roder et al. 2015 style).
    Range [-1, 1]; higher is more interpretable. Returns per-topic values as well."""
    sets = [set(d) for d in docs_tokens]
    n = len(sets)
    vocab = {w for t in topics for w in t[:top_k]}
    doc_freq = {w: 0 for w in vocab}
    for s in sets:
        for w in vocab & s:
            doc_freq[w] += 1
    pair_cache: dict[tuple[str, str], int] = {}

    def co(a: str, b: str) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in pair_cache:
            pair_cache[key] = sum(1 for s in sets if a in s and b in s)
        return pair_cache[key]

    per = []
    for t in topics:
        ws = t[:top_k]
        vals = []
        for a, b in combinations(ws, 2):
            pa, pb = doc_freq[a] / n, doc_freq[b] / n
            pab = co(a, b) / n
            if pab == 0:
                vals.append(-1.0)
            else:
                vals.append(math.log((pab + eps) / (pa * pb + eps)) / -math.log(pab + eps))
        per.append(float(np.mean(vals)) if vals else 0.0)
    return {"mean": float(np.mean(per)) if per else 0.0, "per_topic": per}


def clustering_metrics(X, labels, sample: int = 8000, seed: int = 42) -> dict:
    """Internal validity indices. Undefined for < 2 clusters. Silhouette is computed on a seeded
    subsample for large inputs (it is O(n^2))."""
    labels = np.asarray(labels)
    mask = labels >= 0                       # -1 = noise (e.g. BERTopic outliers)
    Xm, lm = X[mask], labels[mask]
    res: dict = {"n_clusters": int(len(set(lm.tolist()))), "noise_fraction": float(1 - mask.mean())}
    if res["n_clusters"] < 2 or len(lm) <= res["n_clusters"]:
        return res | {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    dense = Xm.toarray() if hasattr(Xm, "toarray") else np.asarray(Xm)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lm), min(sample, len(lm)), replace=False)
    sub_l = lm[idx]
    sil = (float(skm.silhouette_score(dense[idx], sub_l))
           if len(set(sub_l.tolist())) > 1 else None)
    res["silhouette"] = sil
    res["davies_bouldin"] = float(skm.davies_bouldin_score(dense, lm))
    res["calinski_harabasz"] = float(skm.calinski_harabasz_score(dense, lm))
    return res
