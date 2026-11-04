"""Experiment stages used by the Colab notebook: topic benchmark, keyword benchmark, model export,
reference distributions. Every expensive step is a checkpointed stage, so a disconnect loses at most
the method that was running."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import __version__
from .checkpoint import CheckpointStore
from .nlp import keywords as kw
from .nlp import topics as tp


def topic_benchmark(texts: list[str], store: CheckpointStore, methods: list[str], ks: tuple[int, ...],
                    seed: int = 42, encoder=None, max_docs: int = 20000, corpus_name: str = "corpus") -> dict:
    """Compare topic methods over a k sweep on ONE corpus. Embedding-based methods need ``encoder``;
    without it they are reported in ``skipped`` (not silently dropped).

    Returns {'table': DataFrame (one row per method x k), 'best': {method: row}, 'topics': {...},
    'skipped': {...}, 'n_docs': int}."""
    rng = np.random.default_rng(seed)
    if len(texts) > max_docs:
        texts = [texts[i] for i in rng.choice(len(texts), max_docs, replace=False)]
    docs, toks, keep = tp.prepare_docs(texts)
    raw = [texts[i] for i in keep]
    space = tp.evaluation_space(docs, seed)
    needs_enc = {"embedding_kmeans", "bertopic"}
    emb = None
    skipped: dict[str, str] = {}
    if needs_enc & set(methods):
        if encoder is None:
            for m in needs_enc & set(methods):
                skipped[m] = "no sentence encoder available (install sentence-transformers)"
        else:
            def make_emb():
                store.save_object(f"topics/{corpus_name}/embeddings.pkl", np.asarray(encoder(raw)))
            store.run(f"topics:{corpus_name}:embeddings", make_emb, outputs=[f"topics/{corpus_name}/embeddings.pkl"],
                      params={"n": len(raw), "seed": seed})
            emb = store.load_object(f"topics/{corpus_name}/embeddings.pkl")
    rows, topics_out = [], {}
    for m in methods:
        if m in skipped:
            continue
        for k in ks:
            rel = f"topics/{corpus_name}/{m}_k{k}.json"

            def work(m=m, k=k, rel=rel):
                fn = tp.METHODS.get(m) or {"embedding_kmeans": tp.fit_embedding_kmeans,
                                           "bertopic": tp.fit_bertopic}[m]
                extra = {"encoder": (lambda _t: emb), "texts": raw} if m == "embedding_kmeans" else (
                    {"embeddings": emb} if m == "bertopic" else {})
                res = fn(docs, toks, k, seed=seed, **extra)
                sc = tp.score_result(res, toks, space)
                reps = tp.representative_docs(res, raw, space, 2)
                store.save_json(rel, {**{kk: v for kk, v in sc.items()}, "topics": res.topics,
                                      "sizes": np.bincount(res.labels[res.labels >= 0],
                                                           minlength=len(res.topics)).tolist(),
                                      "representatives": {str(a): b for a, b in reps.items()}})
            try:
                store.run(f"topics:{corpus_name}:{m}:k{k}", work, outputs=[rel],
                          params={"m": m, "k": k, "seed": seed, "n": len(docs), "v": __version__})
                d = store.load_json(rel)
                rows.append({kk: d[kk] for kk in ("method", "k", "npmi_coherence", "topic_diversity", "silhouette",
                                                  "davies_bouldin", "calinski_harabasz", "noise_fraction",
                                                  "seconds")})
                topics_out[(m, k)] = d
            except Exception as exc:        # noqa: BLE001
                skipped[f"{m}:k{k}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    table = pd.DataFrame(rows)
    best = {}
    if len(table):
        for m, g in table.groupby("method"):
            best[m] = tp.best_k(g.to_dict("records"))
    return {"table": table, "best": best, "topics": topics_out, "skipped": skipped, "n_docs": len(docs)}


def keyword_benchmark(texts: list[str], store: CheckpointStore, encoder=None, top_n: int = 20, seed: int = 42,
                      name: str = "corpus", max_texts: int = 10000) -> dict:
    texts = texts[:max_texts]
    rel = f"keywords/{name}.pkl"

    def work():
        store.save_object(rel, kw.run_all(texts, top_n, None, encoder, stability_runs=3, seed=seed))
    store.run(f"keywords:{name}", work, outputs=[rel], params={"n": len(texts), "top_n": top_n, "seed": seed,
                                                              "enc": encoder is not None, "v": __version__})
    res = store.load_object(rel)
    summary = pd.DataFrame([{"method": m, "seconds": res["seconds"][m],
                             "stability_jaccard": res["stability_jaccard"][m],
                             "top_keywords": ", ".join(k for k, _ in res["keywords"][m][:10])}
                            for m in res["keywords"]])
    return {"summary": summary, "raw": res}


def reference_distribution(store: CheckpointStore, task: str, method: str) -> dict[str, float]:
    """Predicted-label distribution of a method on the held-out test set (reference for drift/PSI)."""
    from .nlp.benchmark import _safe
    d = store.load_object(f"nlp/{task}/{_safe(method)}/test_pred.pkl")
    vc = pd.Series(d["pred"]).value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def best_sklearn_method(table: pd.DataFrame) -> str | None:
    sk = table[table["family"] == "tfidf"]
    return None if sk.empty else str(sk.sort_values("macro_f1", ascending=False).iloc[0]["method"])


def export_model(store: CheckpointStore, task: str, method: str, out_dir, filename: str, metrics: dict,
                 data_fingerprint: str) -> str:
    """Copy a verified sklearn pipeline to ``out_dir`` with a model card (what it is, how good, how it was
    trained). Refuses if the checkpoint does not verify."""
    import json
    from pathlib import Path

    from .checkpoint import atomic_write_bytes, atomic_write_json
    from .nlp.benchmark import _safe
    ok, why = store.verify(f"{task}:{method}")
    if not ok:
        raise RuntimeError(f"cannot export {method}: {why}")
    src = store.root / f"nlp/{task}/{_safe(method)}/model.joblib"
    out = Path(out_dir)
    atomic_write_bytes(out / filename, src.read_bytes())
    card = {"task": task, "method": method, "chatlens_version": __version__, "dataset_fingerprint": data_fingerprint,
            "test_macro_f1": metrics["macro_f1"], "test_macro_f1_ci95": metrics["macro_f1_ci95"],
            "test_accuracy": metrics["accuracy"], "classes": metrics["confusion_matrix"]["labels"],
            "intended_use": "Sentiment of Roman-Urdu/Hinglish chat messages (exploratory analytics).",
            "limitations": "Trained on e-commerce/social text; not validated on WhatsApp conversations; "
                           "sarcasm and code-switching degrade accuracy."}
    atomic_write_json(out / (Path(filename).stem + ".card.json"), card)
    json.loads((out / (Path(filename).stem + ".card.json")).read_text())
    return str(out / filename)


def prepare_dataset(store: CheckpointStore, name: str, loader, presplit: bool = False, seed: int = 42,
                    max_rows: int | None = None) -> dict | None:
    """Checkpointed load -> clean -> split of a labelled corpus.

    ``loader()`` returns a DataFrame(text,label) or, if ``presplit``, a dict of train/val/test frames
    (official splits are respected; leakage between them is still checked and removed from val/test)."""
    from .kaggle_data import assert_no_leakage, clean_labeled, split_dataset, text_key
    from .tracking import dataset_fingerprint
    rels = [f"data/processed/{name}/{s}.pkl" for s in ("train", "val", "test")] + [
        f"data/processed/{name}/quality.json"]

    def work():
        raw = loader()
        if presplit:
            parts, reports = {}, {}
            for k, d in raw.items():
                parts[k], reports[k] = clean_labeled(d)
            seen = set(parts["train"]["text"].map(text_key))
            leaked = {}
            for k in ("val", "test"):
                keys = parts[k]["text"].map(text_key)
                mask = keys.isin(seen)
                leaked[k] = int(mask.sum())
                parts[k] = parts[k][~mask].reset_index(drop=True)
                seen |= set(parts[k]["text"].map(text_key))
            assert_no_leakage(parts)
            quality = {"presplit": True, "per_split": {k: r.to_dict() for k, r in reports.items()},
                       "removed_for_leakage": leaked}
            splits = parts
        else:
            df, rep = clean_labeled(raw)
            n_before = len(df)
            if max_rows and len(df) > max_rows:
                df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
            splits = split_dataset(df, seed)
            quality = {"presplit": False, **rep.to_dict(), "subsampled_from": n_before if len(df) < n_before else None,
                       "class_counts_used": df["label"].value_counts().to_dict()}
        quality["sizes"] = {k: len(v) for k, v in splits.items()}
        quality["fingerprint"] = dataset_fingerprint(__import__("pandas").concat(list(splits.values())))
        for k, v in splits.items():
            store.save_df(f"data/processed/{name}/{k}.pkl", v)
        store.save_json(f"data/processed/{name}/quality.json", quality)

    store.run(f"prepare:{name}", work, outputs=rels, params={"name": name, "seed": seed, "max_rows": max_rows,
                                                             "v": __version__})
    splits = {k: store.load_df(f"data/processed/{name}/{k}.pkl") for k in ("train", "val", "test")}
    return {"splits": splits, "quality": store.load_json(f"data/processed/{name}/quality.json")}


def run_classification_task(task: str, prepared: dict, registry, store: CheckpointStore, tracker=None, seed: int = 42,
                            n_boot: int = 1000) -> dict:
    """Benchmark every method, build the comparison + significance tables, log to the tracker."""
    from .nlp import benchmark as bm
    fp = prepared["quality"]["fingerprint"]
    res = bm.run_benchmark(task, prepared["splits"], registry, store, seed, n_boot, data_fingerprint=fp)
    if not res["metrics"]:
        return {**res, "table": pd.DataFrame(), "significance": pd.DataFrame()}
    table = bm.comparison_table(res["metrics"])
    sig = bm.significance_table(task, prepared["splits"]["test"]["label"], store, res["classes"], list(res["metrics"]),
                                n_boot, seed)
    store.save_csv(f"reports/tables/{task}_comparison.csv", table)
    store.save_csv(f"reports/tables/{task}_significance.csv", sig)
    if tracker is not None:
        tracker.log_benchmark(task, res["metrics"], fp, seed, store.root)
    return {**res, "table": table, "significance": sig}
