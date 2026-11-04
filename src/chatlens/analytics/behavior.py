"""D. Conversation behaviour: sessions, initiation, turn-taking, response times.

ASSUMPTIONS (read before interpreting any number):
 1. A *turn* is a run of consecutive messages by the same sender.
 2. A *response* is the first message of a turn measured from the LAST message of the previous
    turn by a different sender. WhatsApp exports carry no reply links or read receipts, so
    "B responded to A" is an inference from adjacency, not a fact. In groups the reply may be
    addressed to someone else entirely.
 3. Gaps longer than ``response_window_minutes`` are NOT counted as responses (treated as a new
    topic or a conversation restart); they are counted separately.
 4. A turn with "no observed reply" means no other participant wrote within the window after it.
    It does NOT mean the message was ignored: the person may have read it, replied by phone call,
    or replied on another channel. The final turn of the export is excluded (the export ends).
 5. A *session* starts after a silence of ``session_gap_minutes`` or more.
 6. Timestamps have minute resolution (WhatsApp), so sub-minute response times are 0-59 s
    quantised; very fast replies are therefore not resolvable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import stats
from ..config import AnalysisConfig
from ._common import user_messages


def add_sessions(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    cfg = config or AnalysisConfig()
    um = user_messages(df).sort_values(["timestamp", "msg_id"]).copy()
    gap = um["timestamp"].diff()
    new = gap.isna() | (gap >= pd.Timedelta(minutes=cfg.session_gap_minutes))
    um["session_id"] = new.cumsum() - 1
    um["is_session_start"] = new
    return um


def turns(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    """Collapse consecutive same-sender messages into turns (with session ids)."""
    um = add_sessions(df, config)
    if um.empty:
        return pd.DataFrame(columns=["turn_id", "sender", "start", "end", "n_messages",
                                     "session_id"])
    block = ((um["sender"] != um["sender"].shift()) | um["is_session_start"]).cumsum()
    t = um.groupby(block).agg(sender=("sender", "first"), start=("timestamp", "min"),
                              end=("timestamp", "max"), n_messages=("msg_id", "size"),
                              session_id=("session_id", "first")).reset_index(drop=True)
    t.insert(0, "turn_id", range(len(t)))
    return t


def sessions_summary(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    um = add_sessions(df, config)
    if um.empty:
        return pd.DataFrame()
    g = um.groupby("session_id")
    s = pd.DataFrame({"start": g["timestamp"].min(), "end": g["timestamp"].max(),
                      "messages": g.size(), "participants": g["sender"].nunique(),
                      "initiator": g["sender"].first()})
    s["duration_min"] = (s["end"] - s["start"]).dt.total_seconds() / 60
    return s


def initiation(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    s = sessions_summary(df, config)
    if s.empty:
        return pd.DataFrame(columns=["sender", "sessions_started", "pct"])
    c = s["initiator"].value_counts().rename_axis("sender").reset_index(name="sessions_started")
    c["pct"] = 100 * c["sessions_started"] / c["sessions_started"].sum()
    return c


def _empty_response_frame() -> pd.DataFrame:
    """Typed empty frame: an object-dtype empty mask would be misread by pandas as column names."""
    return pd.DataFrame({"responder": pd.Series(dtype="object"), "previous_sender": pd.Series(dtype="object"),
                         "minutes": pd.Series(dtype="float64"), "same_session": pd.Series(dtype="bool"),
                         "within_window": pd.Series(dtype="bool")})


def response_times(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    """One row per turn change: responder, previous speaker, minutes. Within-window only."""
    cfg = config or AnalysisConfig()
    t = turns(df, cfg)
    if len(t) < 2:
        return _empty_response_frame()
    prev_end = t["end"].shift()
    prev_sender = t["sender"].shift()
    same_session = t["session_id"] == t["session_id"].shift()
    rt = (t["start"] - prev_end).dt.total_seconds() / 60
    out = pd.DataFrame({"responder": t["sender"], "previous_sender": prev_sender, "minutes": rt,
                        "same_session": same_session}).iloc[1:]
    out = out[out["responder"] != out["previous_sender"]]
    out["within_window"] = (out["minutes"] <= cfg.response_window_minutes).astype(bool)
    return out.reset_index(drop=True)


def response_summary(df: pd.DataFrame, config: AnalysisConfig | None = None) -> dict:
    cfg = config or AnalysisConfig()
    rt = response_times(df, cfg)
    valid = rt[rt["within_window"].astype(bool)]
    res: dict = {"turn_changes": int(len(rt)), "counted_responses": int(len(valid)),
                 "excluded_beyond_window": int((~rt["within_window"].astype(bool)).sum()),
                 "window_minutes": cfg.response_window_minutes, "by_responder": {}}
    if valid.empty:
        return res
    est, lo, hi = stats.bootstrap_ci(valid["minutes"], np.median, cfg.bootstrap_iterations,
                                     cfg.alpha, cfg.seed)
    res["overall"] = {"median_min": est, "median_ci95": (lo, hi),
                      "mean_min": float(valid["minutes"].mean()),
                      "p90_min": float(valid["minutes"].quantile(0.9))}
    for who, g in valid.groupby("responder"):
        if len(g) >= 5:
            e, l_, h_ = stats.bootstrap_ci(g["minutes"], np.median, cfg.bootstrap_iterations,
                                           cfg.alpha, cfg.seed)
            res["by_responder"][who] = {"n": int(len(g)), "median_min": e,
                                        "median_ci95": (l_, h_)}
    k = stats.kruskal_by_group(valid["minutes"], valid["responder"], cfg.alpha)
    res["difference_between_responders"] = k.to_dict() if k else None
    return res


def response_distribution(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    """Binned within-window response times for plotting."""
    rt = response_times(df, config)
    v = rt.loc[rt["within_window"].astype(bool), "minutes"]
    bins = [-0.001, 1, 5, 15, 60, 180, 360, np.inf]
    labels = ["<=1 min", "1-5 min", "5-15 min", "15-60 min", "1-3 h", "3-6 h", ">6 h"]
    return pd.cut(v, bins, labels=labels).value_counts().reindex(labels, fill_value=0) \
        .rename_axis("bucket").reset_index(name="responses")


def no_observed_reply(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    """Turns with no reply by another participant within the window (see assumption 4)."""
    cfg = config or AnalysisConfig()
    t = turns(df, cfg)
    if len(t) < 2:
        return pd.DataFrame(columns=["sender", "turns", "no_observed_reply", "pct"])
    nxt_sender, nxt_start = t["sender"].shift(-1), t["start"].shift(-1)
    gap = (nxt_start - t["end"]).dt.total_seconds() / 60
    replied = (nxt_sender != t["sender"]) & (gap <= cfg.response_window_minutes)
    body = t.iloc[:-1].assign(replied=replied.iloc[:-1])   # final turn excluded
    g = body.groupby("sender")["replied"]
    res = pd.DataFrame({"turns": g.size(), "no_observed_reply": g.size() - g.sum()})
    res["pct"] = 100 * res["no_observed_reply"] / res["turns"]
    return res.sort_values("no_observed_reply", ascending=False).reset_index()


def interaction_matrix(df: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    """Counts of 'row responds after column' adjacency pairs within the response window."""
    rt = response_times(df, config)
    rt = rt[rt["within_window"].astype(bool)]
    return pd.crosstab(rt["responder"], rt["previous_sender"])


def exchange_stats(df: pd.DataFrame, config: AnalysisConfig | None = None) -> dict:
    t = turns(df, config)
    if t.empty:
        return {}
    per_session = t.groupby("session_id").size()
    um = add_sessions(df, config)
    days = max((um["timestamp"].max() - um["timestamp"].min()).days + 1, 1)
    return {"sessions": int(um["session_id"].nunique()),
            "avg_turns_per_session": float(per_session.mean()),
            "median_turns_per_session": float(per_session.median()),
            "back_and_forth_sessions": int((per_session >= 4).sum()),
            "turn_changes_per_day": float((len(t) - um["session_id"].nunique()) / days)}
