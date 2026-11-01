"""A. Overall conversation analytics."""
from __future__ import annotations

import pandas as pd

from ._common import text_messages, user_messages


def overview(df: pd.DataFrame, top_n: int = 10, include_text: bool = False) -> dict:
    """Headline numbers. Longest/shortest messages return length + row id by default; pass
    ``include_text=True`` only when showing content to the data owner."""
    um = user_messages(df)
    tm = text_messages(df)
    res: dict = {
        "total_messages": int((~df["is_system"]).sum()),
        "system_messages": int(df["is_system"].sum()),
        "total_participants": int(um["sender"].nunique()),
        "total_words": int(tm["n_words"].sum()),
        "total_characters": int(tm["n_chars"].sum()),
        "avg_chars_per_message": float(tm["n_chars"].mean()) if len(tm) else 0.0,
        "avg_words_per_message": float(tm["n_words"].mean()) if len(tm) else 0.0,
        "media_count": int(um["is_media"].sum()),
        "link_count": int(tm["urls"].str.len().sum()),
        "messages_with_links": int((tm["urls"].str.len() > 0).sum()),
        "location_shares": int(um["is_location"].sum()),
        "deleted_messages": int(um["is_deleted"].sum()),
        "date_start": um["timestamp"].min(), "date_end": um["timestamp"].max(),
        "span_days": int((um["timestamp"].max() - um["timestamp"].min()).days + 1)
        if len(um) else 0,
    }
    res["top_participants"] = (um["sender"].value_counts().head(top_n)
                               .rename_axis("sender").reset_index(name="messages"))
    for label, row in (("longest_message", tm.loc[tm["n_chars"].idxmax()] if len(tm) else None),
                       ("shortest_message", tm.loc[tm["n_chars"].idxmin()] if len(tm) else None)):
        if row is not None:
            item = {"msg_id": int(row["msg_id"]), "sender": row["sender"],
                    "n_chars": int(row["n_chars"]), "n_words": int(row["n_words"])}
            if include_text:
                item["text"] = row["message"]
            res[label] = item
    res["growth_daily"] = _growth(um, "D")
    res["growth_monthly"] = _growth(um, "MS")
    return res


def _growth(um: pd.DataFrame, freq: str) -> pd.DataFrame:
    if um.empty:
        return pd.DataFrame(columns=["period", "messages", "cumulative"])
    s = um.set_index("timestamp").resample(freq).size()
    return pd.DataFrame({"period": s.index, "messages": s.to_numpy(),
                         "cumulative": s.cumsum().to_numpy()})
