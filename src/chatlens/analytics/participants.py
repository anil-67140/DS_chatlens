"""C. Participant analytics."""
from __future__ import annotations

import pandas as pd

from .. import stats
from ._common import text_messages, user_messages


def participant_table(df: pd.DataFrame) -> pd.DataFrame:
    um = user_messages(df)
    tm = text_messages(df)
    g_all = um.groupby("sender")
    g_txt = tm.groupby("sender")
    tbl = pd.DataFrame({
        "messages": g_all.size(),
        "words": g_txt["n_words"].sum(),
        "chars": g_txt["n_chars"].sum(),
        "avg_chars": g_txt["n_chars"].mean(),
        "avg_words": g_txt["n_words"].mean(),
        "median_words": g_txt["n_words"].median(),
        "media": g_all["is_media"].sum(),
        "links": g_txt["urls"].apply(lambda s: int(s.str.len().sum())),
        "emojis": g_txt["n_emojis"].sum(),
        "first_message": g_all["timestamp"].min(),
        "last_message": g_all["timestamp"].max(),
    }).fillna(0)
    tbl["pct_of_messages"] = 100 * tbl["messages"] / tbl["messages"].sum()
    tbl["pct_of_words"] = 100 * tbl["words"] / tbl["words"].sum() if tbl["words"].sum() else 0.0
    tbl["emojis_per_100_words"] = 100 * tbl["emojis"] / tbl["words"].replace(0, float("nan"))
    tbl["active_days"] = g_all["timestamp"].apply(lambda s: s.dt.normalize().nunique())
    return tbl.sort_values("messages", ascending=False)


def concentration(df: pd.DataFrame) -> dict:
    """Participation inequality. Gini 0 = everyone equal, ->1 = one person dominates."""
    c = user_messages(df)["sender"].value_counts()
    if c.empty:
        return {}
    s = c / c.sum()
    return {"participants": int(len(c)), "gini": stats.gini(c.to_numpy()),
            "hhi": stats.hhi(c.to_numpy()),
            "normalized_entropy": stats.normalized_entropy(c.to_numpy()),
            "top1_share": float(s.iloc[0]), "top3_share": float(s.head(3).sum()),
            "equal_share": float(1 / len(c)),
            "note": "Gini/HHI of 2-person chats mainly reflect who talks slightly more."}


def monthly_activity(df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """Messages per month for the top-k participants (others pooled), wide format."""
    um = user_messages(df)
    if um.empty:
        return pd.DataFrame()
    top = um["sender"].value_counts().head(top_k).index
    s = um.assign(who=um["sender"].where(um["sender"].isin(top), "Others"),
                  month=um["timestamp"].dt.to_period("M").dt.to_timestamp())
    return s.pivot_table(index="month", columns="who", values="msg_id", aggfunc="count",
                         fill_value=0)


def compare_participants(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Do participants differ in message length and in hour-of-day habits?
    Kruskal-Wallis (+epsilon^2) and chi-square (+Cramer's V). Effect sizes are reported
    because with thousands of messages even trivial differences are 'significant'."""
    tm = text_messages(df)
    tm = tm[tm["timestamp"].notna()]
    out: dict = {}
    k = stats.kruskal_by_group(tm["n_words"], tm["sender"], alpha)
    out["message_length_by_sender"] = k.to_dict() if k else None
    top = tm["sender"].value_counts()
    top = top[top >= 30].index
    sub = tm[tm["sender"].isin(top)]
    bucket = pd.cut(sub["timestamp"].dt.hour, [-1, 5, 11, 17, 23],
                    labels=["night", "morning", "afternoon", "evening"])
    c = stats.chi_square_independence(sub["sender"], bucket, alpha)
    out["time_of_day_by_sender"] = c.to_dict() if c else None
    return out
