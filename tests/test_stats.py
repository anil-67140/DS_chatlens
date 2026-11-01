import numpy as np
import pandas as pd
import pytest

from chatlens import stats


def test_gini_extremes():
    assert stats.gini([5, 5, 5, 5]) == pytest.approx(0)
    assert stats.gini([0, 0, 0, 100]) == pytest.approx(0.75)  # (n-1)/n for n=4


def test_entropy_and_hhi():
    assert stats.normalized_entropy([1, 1, 1, 1]) == pytest.approx(1)
    assert stats.hhi([1, 1, 1, 1]) == pytest.approx(0.25)
    assert stats.normalized_entropy([10]) == 0


def test_bootstrap_is_seeded_and_covers_truth():
    x = np.random.default_rng(1).normal(10, 2, 500)
    a = stats.bootstrap_ci(x, np.median, 500, seed=7)
    b = stats.bootstrap_ci(x, np.median, 500, seed=7)
    assert a == b and a[1] < 10 < a[2]


def test_kruskal_detects_planted_difference_with_effect_size():
    rng = np.random.default_rng(0)
    v = pd.Series(np.r_[rng.normal(5, 1, 100), rng.normal(9, 1, 100)])
    g = pd.Series(["a"] * 100 + ["b"] * 100)
    r = stats.kruskal_by_group(v, g)
    assert r.significant and r.magnitude == "large"


def test_significant_but_negligible_effect_is_labelled_negligible():
    rng = np.random.default_rng(0)
    a = rng.normal(0.00, 1, 200_000)
    b = rng.normal(0.02, 1, 200_000)
    r = stats.mann_whitney(a, b)
    assert r.significant and r.magnitude == "negligible"


def test_chi_square_independence_and_sparse_warning():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.choice(["a", "b"], 500))
    y = pd.Series(rng.choice(["p", "q", "r"], 500))
    r = stats.chi_square_independence(x, y)
    assert not r.significant and r.magnitude == "negligible"
    sparse = stats.chi_square_independence(pd.Series(list("aabb")), pd.Series(list("pqpq")))
    assert "cautiously" in sparse.note


def test_degenerate_inputs_return_none():
    assert stats.kruskal_by_group(pd.Series([1, 2]), pd.Series(["a", "a"])) is None
    assert stats.spearman([1, 1, 1], [1, 2, 3]) is None
    assert stats.theil_sen([1, 2]) is None


def test_theil_sen_recovers_slope():
    s, lo, hi = stats.theil_sen(2 * np.arange(30) + 1)
    assert s == pytest.approx(2) and lo <= 2 <= hi


def test_large_n_median_ci_is_cheap_and_agrees_with_bootstrap():
    x = np.random.default_rng(3).exponential(10, 20_000)
    est, lo, hi = stats.bootstrap_ci(x, np.median, 300)            # order-statistic path
    _, blo, bhi = stats.bootstrap_ci(x, np.median, 300, large_n=10**9)  # forced bootstrap
    assert lo <= est <= hi
    assert abs(lo - blo) < 0.15 and abs(hi - bhi) < 0.15


def test_bootstrap_memory_is_bounded():
    # 1.2M-element index matrix would be required unrolled; blocks keep it under the cap
    x = np.random.default_rng(0).normal(size=4000)
    est, lo, hi = stats.bootstrap_ci(x, np.mean, 300, max_block_elements=100_000)
    assert lo < est < hi
