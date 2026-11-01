"""B. Temporal analytics."""
from __future__ import annotations

import pandas as pd

from .. import stats
from ._common import WEEKDAYS, user_messages


def hourly(df: pd.DataFrame) -> pd.Series:
    um = user_messages(df)
    return um["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0).sort_index()


def weekday(df: pd.DataFrame) -> pd.Series:
    um = user_messages(df)
    return um["timestamp"].dt.day_name().value_counts().reindex(WEEKDAYS, fill_value=0)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    um = user_messages(df)
    if um.empty:
        return pd.DataFrame(columns=["month", "messages", "pct_change"])
    s = um.set_index("timestamp").resample("MS").size()
    return pd.DataFrame({"month": s.index, "messages": s.to_numpy(),
                         "pct_change": s.pct_change().replace([float("inf")], float("nan"))
                         .to_numpy()})


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    um = user_messages(df)
    if um.empty:
        return pd.DataFrame(columns=["week", "messages"])
    s = um.set_index("timestamp").resample("W-MON", label="left", closed="left").size()
    return pd.DataFrame({"week": s.index, "messages": s.to_numpy()})


def heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Weekday x hour message counts."""
    um = user_messages(df)
    t = pd.crosstab(um["timestamp"].dt.day_name(), um["timestamp"].dt.hour)
    return t.reindex(index=WEEKDAYS, columns=range(24), fill_value=0)


def daily_counts(df: pd.DataFrame) -> pd.Series:
    """Messages per calendar day INCLUDING zero-message days (needed for honest averages)."""
    um = user_messages(df)
    if um.empty:
        return pd.Series(dtype=int)
    s = um.set_index("timestamp").resample("D").size()
    return s


def weekend_vs_weekday(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    d = daily_counts(df)
    if d.empty:
        return {}
    weekend = d[d.index.dayofweek >= 5]
    week = d[d.index.dayofweek < 5]
    test = stats.mann_whitney(weekend.to_numpy(), week.to_numpy(), alpha)
    return {"weekend_mean_per_day": float(weekend.mean()) if len(weekend) else float("nan"),
            "weekday_mean_per_day": float(week.mean()) if len(week) else float("nan"),
            "weekend_days": int(len(weekend)), "weekday_days": int(len(week)),
            "test": test.to_dict() if test else None}


def top_hours(df: pd.DataFrame, n: int = 3) -> list[int]:
    return hourly(df).sort_values(ascending=False).head(n).index.tolist()


def longest_inactive_periods(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Largest silences between consecutive user messages."""
    um = user_messages(df).sort_values("timestamp")
    gap = um["timestamp"].diff()
    out = pd.DataFrame({"silence_start": um["timestamp"].shift(1), "silence_end": um["timestamp"],
                        "gap": gap}).dropna()
    return out.sort_values("gap", ascending=False).head(n).reset_index(drop=True)


def streaks(df: pd.DataFrame) -> dict:
    """Consecutive calendar days with >= 1 message."""
    d = daily_counts(df)
    if d.empty:
        return {"longest_days": 0, "longest_start": None, "longest_end": None,
                "current_days": 0}
    active = (d > 0).to_numpy()
    best = cur = 0
    best_end = -1
    for i, a in enumerate(active):
        cur = cur + 1 if a else 0
        if cur > best:
            best, best_end = cur, i
    return {"longest_days": int(best),
            "longest_start": d.index[best_end - best + 1] if best else None,
            "longest_end": d.index[best_end] if best else None,
            "current_days": int(cur)}


def trend(df: pd.DataFrame) -> dict:
    """Robust trend of weekly volume (Theil-Sen slope, messages/week per week) plus Spearman."""
    w = weekly(df)
    if len(w) < 8:
        return {"reliable": False, "reason": "fewer than 8 weeks of data"}
    ts = stats.theil_sen(w["messages"])
    sp = stats.spearman(range(len(w)), w["messages"])
    return {"reliable": True, "weeks": len(w),
            "theil_sen_slope": ts[0], "slope_ci95": (ts[1], ts[2]),
            "spearman": sp.to_dict() if sp else None}


def seasonality(df: pd.DataFrame) -> dict:
    """Mean messages per day by calendar month. Only labelled reliable with >= 24 months."""
    d = daily_counts(df)
    if d.empty:
        return {"reliable": False, "profile": pd.Series(dtype=float)}
    months_span = (d.index.max().year - d.index.min().year) * 12 + d.index.max().month \
        - d.index.min().month + 1
    return {"reliable": months_span >= 24, "months_span": int(months_span),
            "profile": d.groupby(d.index.month).mean().reindex(range(1, 13)),
            "note": "Needs >= 2 full years to separate seasonality from one-off events."}
