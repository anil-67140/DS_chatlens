import numpy as np
import pandas as pd
import pytest

from chatlens.checkpoint import CheckpointStore
from chatlens.nlp import benchmark
from chatlens.nlp.classifiers import (
    HFPretrainedClassifier,
    LexiconClassifier,
    MajorityBaseline,
    emotion_registry,
    sentiment_registry,
    tfidf_methods,
)
from chatlens.preprocessing.normalize import normalize_for_classification


def toy(n=240, seed=0):
    """SYNTHETIC separable sentiment data for plumbing tests only (not a benchmark)."""
    rng = np.random.default_rng(seed)
    pos = ["great", "love", "wonderful", "happy", "excellent", "nice"]
    neg = ["awful", "hate", "terrible", "sad", "horrible", "bad"]
    neu = ["table", "window", "today", "meeting", "paper", "street"]
    rows = []
    for i in range(n):
        lab = ["positive", "negative", "neutral"][i % 3]
        pool = {"positive": pos, "negative": neg, "neutral": neu}[lab]
        words = rng.choice(pool, 3).tolist() + [f"w{rng.integers(0, 30)}"]
        rows.append((" ".join(words) + f" item{i}", lab))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return {"train": df.iloc[:150].reset_index(drop=True), "val": df.iloc[150:195].reset_index(drop=True),
            "test": df.iloc[195:].reset_index(drop=True)}


def test_normalize_for_classification_keeps_emoji_and_squeezes():
    t = normalize_for_classification("Sooooo GOOD 😍 see https://x.com @bob")
    assert "soo" in t and "sooo" not in t and "emo_" in t and "urltoken" in t and "usertoken" in t
    assert "http" not in t and "@bob" not in t


def test_registry_sizes_and_lazy_imports():
    s = sentiment_registry(include_hf=True)
    names = [m.name for m in s]
    assert len(names) == len(set(names))
    assert len([m for m in s if m.family != "baseline"]) >= 10
    assert {"vader", "textblob", "hf_xlmr_twitter_sentiment", "ft_xlm_roberta_base"} <= set(names)
    e = [m.name for m in emotion_registry()]
    assert "vader" not in e and "textblob" not in e          # polarity tools are not emotion models
    assert len([m for m in emotion_registry() if m.family != "baseline"]) >= 10


def test_majority_baseline():
    sp = toy()
    m = MajorityBaseline().fit(sp["train"])
    assert set(m.predict(sp["test"]["text"])) <= set(m.classes_)
    assert m.predict_proba(["x"]).shape == (1, 3)


def test_vader_polarity_and_neutral_band():
    m = LexiconClassifier("vader")
    m.classes_ = ["negative", "neutral", "positive"]
    assert list(m.predict(["I love this, it is wonderful!", "This is terrible and awful.",
                           "The table is in the room."])) == ["positive", "negative", "neutral"]
    assert m.has_proba is False and m.zero_shot
    with pytest.raises(NotImplementedError):
        m.predict_proba(["x"])


def test_textblob_runs_offline():
    m = LexiconClassifier("textblob")
    m.classes_ = ["negative", "neutral", "positive"]
    assert m.predict(["what a wonderful lovely day"])[0] == "positive"


def test_lexicon_binary_dataset_never_predicts_neutral():
    m = LexiconClassifier("vader")
    m.classes_ = ["negative", "positive"]
    assert set(m.predict(["the table", "great", "awful"])) <= {"negative", "positive"}


def test_tuned_lexicon_learns_a_band():
    df = pd.DataFrame({"text": ["great"] * 30 + ["awful"] * 30 + ["the table"] * 30,
                       "label": ["positive"] * 30 + ["negative"] * 30 + ["neutral"] * 30})
    m = LexiconClassifier("vader", tune=True).fit(df)
    assert "neutral_band" in m.best_params_ and m.zero_shot is False
    assert (m.predict(df["text"]) == df["label"]).mean() > 0.95


def test_all_tfidf_methods_learn_separable_data_and_align_classes():
    sp = toy()
    for m in tfidf_methods():
        m.fit(sp["train"], sp["val"])
        proba = m.predict_proba(sp["test"]["text"])
        assert proba.shape == (len(sp["test"]), 3) and np.allclose(proba.sum(1), 1, atol=1e-6), m.name
        acc = (m.predict(sp["test"]["text"]) == sp["test"]["label"].to_numpy()).mean()
        assert acc > 0.9, (m.name, acc)
        assert m.classes_ == ["negative", "neutral", "positive"]


def test_hyperparameter_selection_uses_validation_not_test():
    sp = toy()
    m = tfidf_methods()[0].fit(sp["train"], sp["val"])
    assert "C" in m.best_params_ and m.val_macro_f1 is not None


class FakePipe:
    """Stands in for a transformers pipeline: positive if 'good' in text, negative if 'bad'."""
    def __init__(self, labels=("POSITIVE", "NEGATIVE")):
        self.labels, self.n = labels, 0

    def __call__(self, texts):
        self.n += len(texts)
        out = []
        for t in texts:
            p = 0.95 if "good" in t else 0.05 if "bad" in t else 0.5
            out.append([{"label": self.labels[0], "score": p}, {"label": self.labels[1], "score": 1 - p}])
        return out


def test_hf_pretrained_label_mapping_and_neutral_threshold_learning():
    df = pd.DataFrame({"text": ["good one"] * 20 + ["bad one"] * 20 + ["so so"] * 20,
                       "label": ["positive"] * 20 + ["negative"] * 20 + ["neutral"] * 20})
    fake = FakePipe()
    m = HFPretrainedClassifier("fake", "x/y", {"POSITIVE": "positive", "NEGATIVE": "negative"},
                               pipeline_factory=lambda: fake).fit(df)
    assert m.tau is not None and m.tau > 0
    pred = m.predict(["good one", "bad one", "so so"])
    assert list(pred) == ["positive", "negative", "neutral"]
    p = m.predict_proba(["good one", "so so"])
    assert np.allclose(p.sum(1), 1)


def test_hf_pretrained_model_missing_a_class_is_reported():
    df = pd.DataFrame({"text": ["a", "b", "c"] * 5, "label": ["joy", "anger", "love"] * 5})
    m = HFPretrainedClassifier("fake", "x/y", {"JOY": "joy", "ANGER": "anger"},
                               pipeline_factory=lambda: FakePipe(("JOY", "ANGER"))).fit(df)
    assert m.best_params_ == {"classes_never_predicted": ["love"]}
    assert m.predict_proba(["a"]).shape == (1, 3)


def test_benchmark_runs_checkpoints_and_resumes(tmp_path):
    sp = toy()
    store = CheckpointStore(tmp_path)
    methods = [MajorityBaseline(), LexiconClassifier("vader"), tfidf_methods()[0]]
    r1 = benchmark.run_benchmark("toy", sp, methods, store, n_boot=50)
    assert set(r1["metrics"]) == {"majority_baseline", "vader", "tfidf_word_logreg"}
    assert r1["metrics"]["vader"]["has_probabilities"] is False
    assert "roc_auc_macro_ovr" in r1["metrics"]["tfidf_word_logreg"]
    assert (tmp_path / "nlp/toy/tfidf_word_logreg/model.joblib").exists()

    class Boom(MajorityBaseline):
        def _fit(self, *a):
            raise AssertionError("must not refit a verified method")
    methods2 = [Boom(), LexiconClassifier("vader"), tfidf_methods()[0]]
    r2 = benchmark.run_benchmark("toy", sp, methods2, CheckpointStore(tmp_path), n_boot=50)
    assert r2["errors"] == {} and r2["metrics"].keys() == r1["metrics"].keys()
    tab = benchmark.comparison_table(r2["metrics"])
    assert tab["rank"].tolist() == [1, 2, 3] and tab.iloc[-1]["method"] == "majority_baseline"
    assert tab["macro_f1"].is_monotonic_decreasing
    sig = benchmark.significance_table("toy", sp["test"]["label"], CheckpointStore(tmp_path),
                                       r2["classes"], list(r2["metrics"]), n_boot=50)
    assert {"holm_p", "verdict"} <= set(sig.columns)


def test_one_failing_method_does_not_stop_the_benchmark(tmp_path):
    class Broken(MajorityBaseline):
        name = "broken"

        def _fit(self, *a):
            raise RuntimeError("model download failed")
    r = benchmark.run_benchmark("toy2", toy(), [Broken(), MajorityBaseline()],
                                CheckpointStore(tmp_path), n_boot=20)
    assert "broken" in r["errors"] and "majority_baseline" in r["metrics"]
