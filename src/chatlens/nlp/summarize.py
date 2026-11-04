"""Extractive conversation summaries: select the messages that best represent each session.

Method: TF-IDF centroid similarity with Maximal Marginal Relevance (no generation, so nothing is
invented; every summary line is a real message). This is *extractive*; it is not an abstractive
LLM summary and cannot paraphrase. Only sessions with enough text are summarised."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..analytics import behavior
from ..config import AnalysisConfig
from ..preprocessing.normalize import tokenize


def extractive_summary(messages: list[str], k: int = 3, mmr_lambda: float = 0.7, min_words: int = 3) -> list[int]:
    """Indices (chronological) of the ``k`` most representative, non-redundant messages."""
    eligible = [i for i, m in enumerate(messages) if len(m.split()) >= min_words]
    if len(eligible) <= k:
        return eligible
    try:
        X = TfidfVectorizer(sublinear_tf=True, tokenizer=tokenize, token_pattern=None).fit_transform(
            [messages[i] for i in eligible])
    except ValueError:                      # empty vocabulary
        return eligible[:k]
    cen = np.asarray(X.mean(0)).ravel()
    cen /= np.linalg.norm(cen) + 1e-12
    rel = X @ cen
    rel = np.asarray(rel).ravel()
    chosen: list[int] = []
    while len(chosen) < k:
        rest = [i for i in range(len(eligible)) if i not in chosen]
        if chosen:
            red = np.asarray((X[rest] @ X[chosen].T).max(1).todense()).ravel()
        else:
            red = np.zeros(len(rest))
        score = mmr_lambda * rel[rest] - (1 - mmr_lambda) * red
        chosen.append(rest[int(np.argmax(score))])
    return sorted(eligible[i] for i in chosen)


def summarize_chat(df: pd.DataFrame, config: AnalysisConfig | None = None, top_sessions: int = 5, k: int = 3,
                   include_text: bool = False) -> list[dict]:
    """Summaries of the ``top_sessions`` largest conversation sessions. Text is omitted unless
    ``include_text`` (message ids are always returned)."""
    um = behavior.add_sessions(df, config)
    if um.empty:
        return []
    um = um[um["is_text"]]
    sizes = um.groupby("session_id").size().sort_values(ascending=False)
    out = []
    for sid in sizes.head(top_sessions).index:
        g = um[um["session_id"] == sid].sort_values("timestamp")
        idx = extractive_summary(g["message"].tolist(), k)
        item = {"session_id": int(sid), "start": g["timestamp"].min(), "end": g["timestamp"].max(),
                "n_messages": int(len(g)), "participants": int(g["sender"].nunique()),
                "summary_msg_ids": [int(g.iloc[i]["msg_id"]) for i in idx]}
        if include_text:
            item["summary"] = [g.iloc[i]["message"] for i in idx]
        out.append(item)
    return out
