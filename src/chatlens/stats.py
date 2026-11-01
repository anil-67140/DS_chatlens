"""Statistical helpers. Every test returns the p-value AND an effect size with a plain-language
magnitude, because statistical significance is not practical significance (large chats make tiny
differences "significant")."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as ss


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float | None
    effect_name: str | None
    magnitude: str | None
    significant: bool
    n: int
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def hhi(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.sum() == 0:
        return 0.0
    s = x / x.sum()
    return float((s ** 2).sum())


def normalized_entropy(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[x > 0]
    if x.size <= 1:
        return 0.0
    p = x / x.sum()
    return float(-(p * np.log(p)).sum() / np.log(x.size))


def _mag(value: float, cuts: tuple[float, float, float]) -> str:
    v = abs(value)
    return ("negligible" if v < cuts[0] else "small" if v < cuts[1]
            else "medium" if v < cuts[2] else "large")


def _median_ci_order_statistics(x: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Distribution-free CI for the median from binomial order statistics (no resampling,
    O(n log n), exact up to a one-rank convention)."""
    xs = np.sort(x)
    n = xs.size
    lo = int(ss.binom.ppf(alpha / 2, n, 0.5))
    hi = int(ss.binom.ppf(1 - alpha / 2, n, 0.5))
    return float(np.median(xs)), float(xs[max(lo - 1, 0)]), float(xs[min(hi, n - 1)])


def bootstrap_ci(values, stat=np.median, iterations: int = 2000, alpha: float = 0.05,
                 seed: int = 42, large_n: int = 5000,
                 max_block_elements: int = 5_000_000) -> tuple[float, float, float]:
    """Confidence interval for ``stat``. Returns (estimate, low, high).

    * median with n > ``large_n``: exact distribution-free order-statistic interval (a bootstrap
      would need iterations x n memory and add nothing at that sample size);
    * otherwise: seeded percentile bootstrap, computed in blocks so memory stays bounded.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    if stat is np.median and x.size > large_n:
        return _median_ci_order_statistics(x, alpha)
    rng = np.random.default_rng(seed)
    block = max(1, max_block_elements // x.size)
    boots: list[np.ndarray] = []
    done = 0
    while done < iterations:
        m = min(block, iterations - done)
        idx = rng.integers(0, x.size, size=(m, x.size))
        boots.append(stat(x[idx], axis=1))
        done += m
    lo, hi = np.percentile(np.concatenate(boots), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def kruskal_by_group(values: pd.Series, groups: pd.Series, alpha: float = 0.05,
                     min_group_size: int = 5) -> TestResult | None:
    """Kruskal-Wallis H with epsilon-squared effect size (non-parametric: message lengths and
    response times are heavy-tailed, so ANOVA's normality assumption is not met)."""
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    sizes = d.groupby("g")["v"].size()
    keep = sizes[sizes >= min_group_size].index
    d = d[d["g"].isin(keep)]
    if d["g"].nunique() < 2:
        return None
    samples = [g["v"].to_numpy() for _, g in d.groupby("g")]
    if all(np.ptp(s) == 0 for s in samples) and len({s[0] for s in samples}) == 1:
        return None
    h, p = ss.kruskal(*samples)
    n, k = len(d), len(samples)
    eps2 = float(max(0.0, (h - k + 1) / (n - k))) if n > k else 0.0
    return TestResult("kruskal_wallis", float(h), float(p), eps2, "epsilon_squared",
                      _mag(eps2, (0.01, 0.08, 0.26)), bool(p < alpha), n,
                      f"{k} groups with >= {min_group_size} observations")


def mann_whitney(a, b, alpha: float = 0.05) -> TestResult | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return None
    u, p = ss.mannwhitneyu(a, b, alternative="two-sided")
    rbc = float(2 * u / (a.size * b.size) - 1)  # rank-biserial correlation
    return TestResult("mann_whitney_u", float(u), float(p), rbc, "rank_biserial",
                      _mag(rbc, (0.1, 0.3, 0.5)), bool(p < alpha), int(a.size + b.size))


def chi_square_independence(x: pd.Series, y: pd.Series, alpha: float = 0.05,
                            min_expected: float = 5.0) -> TestResult | None:
    """Chi-square test of independence with Cramer's V. Skips (returns None) when the
    expected-count assumption fails badly or a dimension is degenerate."""
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return None
    chi2, p, dof, exp = ss.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    frac_low = float((exp < min_expected).mean())
    v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    note = "" if frac_low <= 0.2 else (
        f"{frac_low:.0%} of cells have expected count < {min_expected}: treat p-value cautiously")
    return TestResult("chi_square_independence", float(chi2), float(p), v, "cramers_v",
                      _mag(v, (0.1, 0.3, 0.5)), bool(p < alpha), int(n), note)


def spearman(x, y, alpha: float = 0.05) -> TestResult | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return None
    rho, p = ss.spearmanr(x[mask], y[mask])
    return TestResult("spearman", float(rho), float(p), float(rho), "rho",
                      _mag(rho, (0.1, 0.3, 0.5)), bool(p < alpha), int(mask.sum()))


def theil_sen(y) -> tuple[float, float, float] | None:
    """Robust linear trend (slope, low, high 95% CI) of a series indexed 0..n-1."""
    y = np.asarray(y, float)
    if y.size < 4:
        return None
    s, _, lo, hi = ss.theilslopes(y, np.arange(y.size))
    return float(s), float(lo), float(hi)
