import numpy as np
import pytest
from sklearn.datasets import make_blobs

from chatlens.nlp import evaluation as ev

CL = ["neg", "neu", "pos"]


def test_classification_metrics_hand_checked():
    yt = ["neg", "neg", "neu", "neu", "pos", "pos"]
    yp = ["neg", "neu", "neu", "neu", "pos", "neg"]
    m = ev.classification_metrics(yt, yp, CL)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # neg: tp1 fp1 fn1 -> P=.5 R=.5 F=.5 ; neu: tp2 fp1 fn0 -> P=2/3 R=1 F=.8 ; pos: tp1 fp0 fn1 -> P=1 R=.5 F=2/3
    assert m["per_class"]["neg"]["f1"] == pytest.approx(0.5)
    assert m["per_class"]["neu"]["f1"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert "roc_auc_macro_ovr" not in m and m["has_probabilities"] is False


def test_probability_metrics_only_with_probabilities_and_perfect_case():
    yt = np.array(CL * 20)
    proba = np.eye(3)[[0, 1, 2] * 20]
    m = ev.classification_metrics(yt, yt, CL, proba)
    assert m["roc_auc_macro_ovr"] == 1 and m["pr_auc_macro"] == 1
    assert m["ece"] == pytest.approx(0, abs=1e-9) and m["brier"] == pytest.approx(0, abs=1e-9)


def test_ece_known_value():
    # always 90% confident, right half the time -> |0.5-0.9| = 0.4
    y = np.array([0, 1] * 50)
    proba = np.where(np.eye(2)[np.zeros(100, int)] == 1, 0.9, 0.1)   # always predicts class 0 @0.9
    assert ev.expected_calibration_error(y, proba) == pytest.approx(0.4)


def test_random_scores_give_auc_near_half():
    rng = np.random.default_rng(0)
    yt = rng.choice(CL, 3000)
    p = rng.dirichlet([1, 1, 1], 3000)
    m = ev.classification_metrics(yt, np.array(CL)[p.argmax(1)], CL, p)
    assert abs(m["roc_auc_macro_ovr"] - 0.5) < 0.03


def test_mcnemar_exact_known_p():
    yt = np.zeros(20, int)
    a = np.zeros(20, int)                       # A always right
    b = np.r_[np.ones(10, int), np.zeros(10, int)]  # B wrong on 10
    r = ev.mcnemar(yt, a, b)
    assert r["a_only_correct"] == 10 and r["b_only_correct"] == 0
    assert r["test"] == "exact_binomial" and r["p_value"] == pytest.approx(2 * 0.5 ** 10)


def test_mcnemar_identical_models():
    y = np.array([0, 1, 1, 0])
    assert ev.mcnemar(y, y, y)["p_value"] == 1.0


def test_holm_known_values():
    assert ev.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert all(0 <= p <= 1 for p in ev.holm([0.5, 0.9, 0.99]))


def test_paired_bootstrap_detects_real_difference_and_ci_is_seeded():
    rng = np.random.default_rng(1)
    yt = rng.choice(CL, 600)
    good = np.where(rng.random(600) < 0.9, yt, rng.choice(CL, 600))
    bad = np.where(rng.random(600) < 0.5, yt, rng.choice(CL, 600))
    d = ev.paired_bootstrap_diff(yt, good, bad, CL, iterations=300)
    assert d["excludes_zero"] and d["diff"] > 0.2 and d == ev.paired_bootstrap_diff(
        yt, good, bad, CL, iterations=300)


def test_compare_all_labels_equal_models_indistinguishable():
    rng = np.random.default_rng(2)
    yt = rng.choice(CL, 500)
    a = np.where(rng.random(500) < 0.8, yt, rng.choice(CL, 500))
    rows = ev.compare_all(yt, {"a": a, "a_copy": a.copy(), "bad": rng.choice(CL, 500)},
                          CL, reference="a", iterations=200)
    by = {r["method"]: r for r in rows}
    assert by["a_copy"]["verdict"] == "indistinguishable"
    assert by["bad"]["verdict"] == "reference better"


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(3)
    yt = rng.choice(CL, 400)
    yp = np.where(rng.random(400) < 0.7, yt, rng.choice(CL, 400))
    est, lo, hi = ev.bootstrap_metric_ci(yt, yp, CL, iterations=300)
    assert lo < est < hi


def test_topic_diversity():
    assert ev.topic_diversity([["a", "b"], ["c", "d"]], 2) == 1.0
    assert ev.topic_diversity([["a", "b"], ["a", "b"]], 2) == 0.5


def test_npmi_prefers_cooccurring_words():
    docs = [["cat", "dog", "pet"]] * 30 + [["car", "road", "drive"]] * 30
    good = ev.npmi_coherence([["cat", "dog", "pet"], ["car", "road", "drive"]], docs)
    bad = ev.npmi_coherence([["cat", "road", "pet"], ["car", "dog", "drive"]], docs)
    assert good["mean"] == pytest.approx(1.0) and bad["mean"] == pytest.approx(-1 / 3)  # pairs: -1,+1,-1


def test_clustering_metrics_separated_vs_single_cluster():
    X, y = make_blobs(300, centers=3, cluster_std=0.5, random_state=0)
    m = ev.clustering_metrics(X, y)
    assert m["silhouette"] > 0.7 and m["davies_bouldin"] < 0.5 and m["calinski_harabasz"] > 100
    one = ev.clustering_metrics(X, np.zeros(300, int))
    assert one["silhouette"] is None and one["n_clusters"] == 1
    noisy = ev.clustering_metrics(X, np.where(np.arange(300) < 30, -1, y))
    assert noisy["noise_fraction"] == pytest.approx(0.1)


def test_fast_macro_f1_matches_sklearn_including_missing_classes():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(5)
    for _ in range(20):
        yt = rng.choice(CL, 200, p=[0.6, 0.3, 0.1])
        yp = rng.choice(CL[:2], 200)                       # class 'pos' never predicted
        (a, b), fn = ev._prepare(yt, [yp], CL, "macro_f1")
        assert fn(a, b) == pytest.approx(f1_score(yt, yp, labels=CL, average="macro", zero_division=0))
