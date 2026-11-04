import json

import numpy as np
import pandas as pd
import pytest
from test_nlp_topics_keywords import THEMES, corpus, kw_texts

from chatlens.checkpoint import CheckpointStore
from chatlens.experiments import (
    best_sklearn_method,
    export_model,
    keyword_benchmark,
    reference_distribution,
    topic_benchmark,
)
from chatlens.nlp import benchmark
from chatlens.nlp.classifiers import MajorityBaseline, tfidf_methods


def toy_encoder(texts):
    theme_of = {w: i for i, v in enumerate(THEMES.values()) for w in v.split()}
    out = np.zeros((len(texts), 4)) + 1e-3
    for i, t in enumerate(texts):
        for w in t.split():
            if w in theme_of:
                out[i, theme_of[w]] += 1
    return out


def test_topic_benchmark_runs_resumes_and_reports_skips(tmp_path):
    texts, _ = corpus(70)
    store = CheckpointStore(tmp_path)
    r = topic_benchmark(texts, store, ["lda", "nmf", "kmeans_tfidf", "lsa_kmeans", "embedding_kmeans", "bertopic"],
                        ks=(3, 4, 6), encoder=toy_encoder)
    assert set(r["table"]["method"]) >= {"lda", "nmf", "kmeans_tfidf", "lsa_kmeans", "embedding_kmeans"}
    assert any(k.startswith("bertopic") for k in r["skipped"])          # package absent: reported, not hidden
    assert r["best"]["nmf"]["k"] == 4 and r["best"]["embedding_kmeans"]["k"] == 4
    again = topic_benchmark(texts, CheckpointStore(tmp_path), ["nmf"], ks=(3, 4, 6), encoder=toy_encoder)
    assert len(again["table"]) == 3
    assert CheckpointStore(tmp_path).verify("topics:corpus:nmf:k4")[0]


def test_topic_benchmark_without_encoder_skips_embedding_methods(tmp_path):
    texts, _ = corpus(40)
    r = topic_benchmark(texts, CheckpointStore(tmp_path), ["nmf", "embedding_kmeans"], ks=(4,))
    assert "embedding_kmeans" in r["skipped"] and list(r["table"]["method"]) == ["nmf"]


def test_keyword_benchmark_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path)
    r = keyword_benchmark(kw_texts(), store, name="t")
    assert {"frequency", "tfidf", "rake", "yake", "textrank"} <= set(r["summary"]["method"])
    n_before = store.stages["keywords:t"]["finished"]
    keyword_benchmark(kw_texts(), CheckpointStore(tmp_path), name="t")
    assert CheckpointStore(tmp_path).stages["keywords:t"]["finished"] == n_before     # not recomputed


def test_reference_distribution_and_model_export(tmp_path):
    rows = [(f"great word{i} love", "positive") if i % 3 == 0 else (f"awful word{i} hate", "negative")
            if i % 3 == 1 else (f"table word{i} paper", "neutral") for i in range(300)]
    df = pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=1).reset_index(drop=True)
    sp = {"train": df.iloc[:200], "val": df.iloc[200:250], "test": df.iloc[250:].reset_index(drop=True)}
    store = CheckpointStore(tmp_path / "s")
    res = benchmark.run_benchmark("t", sp, [MajorityBaseline(), tfidf_methods()[0]], store, n_boot=30)
    tab = benchmark.comparison_table(res["metrics"])
    best = best_sklearn_method(tab)
    assert best == "tfidf_word_logreg"
    ref = reference_distribution(store, "t", best)
    assert pytest.approx(sum(ref.values())) == 1 and set(ref) == {"positive", "negative", "neutral"}
    path = export_model(store, "t", best, tmp_path / "models", "m.joblib", res["metrics"][best], "abc123")
    card = json.loads((tmp_path / "models" / "m.card.json").read_text())
    assert card["method"] == best and card["dataset_fingerprint"] == "abc123" and "limitations" in card
    import joblib
    assert joblib.load(path).predict(["great love"])[0] == "positive"
    (tmp_path / "s" / "nlp/t/tfidf_word_logreg/model.joblib").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="cannot export"):
        export_model(CheckpointStore(tmp_path / "s"), "t", best, tmp_path / "m2", "m.joblib", res["metrics"][best], "x")


def test_prepare_dataset_presplit_removes_cross_split_leakage_and_resumes(tmp_path):
    from chatlens.experiments import prepare_dataset
    mk = lambda texts, labs: pd.DataFrame({"text": texts, "label": labs})  # noqa: E731
    tr = mk([f"train sentence {i} about thing" for i in range(60)] + ["shared line here"], ["a", "b"] * 30 + ["a"])
    va = mk([f"val sentence {i} about stuff" for i in range(20)] + ["Shared line here!"], ["a", "b"] * 10 + ["a"])
    te = mk([f"test sentence {i} about items" for i in range(20)], ["a", "b"] * 10)
    calls = []

    def loader():
        calls.append(1)
        return {"train": tr, "val": va, "test": te}
    store = CheckpointStore(tmp_path)
    p = prepare_dataset(store, "toy", loader, presplit=True)
    assert p["quality"]["removed_for_leakage"]["val"] == 1 and p["quality"]["sizes"]["val"] == 20
    assert "Shared" not in " ".join(p["splits"]["val"]["text"])
    prepare_dataset(CheckpointStore(tmp_path), "toy", loader, presplit=True)
    assert len(calls) == 1                                            # resumed from checkpoint


def test_prepare_and_run_classification_task_end_to_end(tmp_path):
    from chatlens.experiments import prepare_dataset, run_classification_task
    from chatlens.nlp.classifiers import LexiconClassifier
    from chatlens.tracking import Tracker
    rows = [(f"great lovely wonderful item{i}", "positive") if i % 2
            else (f"awful terrible horrible item{i}", "negative") for i in range(400)]
    store = CheckpointStore(tmp_path / "s")
    prep = prepare_dataset(store, "toy2", lambda: pd.DataFrame(rows, columns=["text", "label"]))
    assert sum(prep["quality"]["sizes"].values()) == 400 and len(prep["quality"]["fingerprint"]) == 16
    tr = Tracker(tmp_path / "t", use_mlflow=False)
    out = run_classification_task("toy2", prep, [MajorityBaseline(), LexiconClassifier("vader"), tfidf_methods()[0]],
                                  store, tr, n_boot=30)
    assert out["table"].iloc[0]["macro_f1"] >= 0.99 and (tmp_path / "s/reports/tables/toy2_comparison.csv").exists()
    assert len((tmp_path / "t/runs.jsonl").read_text().splitlines()) == 3


def test_figures_and_project_report(tmp_path):
    import matplotlib.pyplot as plt

    from chatlens.report import project_report
    from chatlens.viz import benchmark_figure, confusion_figure
    m = {"confusion_matrix": {"labels": ["a", "b"], "matrix": [[8, 2], [1, 9]]}}
    f1 = confusion_figure(m)
    p = tmp_path / "c.png"
    f1.savefig(p)
    plt.close(f1)
    tab = pd.DataFrame({"method": ["x", "y"], "macro_f1": [0.5, 0.6], "macro_f1_ci_low": [0.4, 0.5],
                        "macro_f1_ci_high": [0.6, 0.7]})
    plt.close(benchmark_figure(tab))
    html = project_report("T <b>", [("S1", tab), ("S2", "text <script>x</script>"), ("S3", ("images", [str(p)])),
                                    ("S4", {"k": "v"})], {"version": "1"})
    assert "<script>x</script>" not in html and "data:image/png;base64" in html and "&lt;b&gt;" in html
