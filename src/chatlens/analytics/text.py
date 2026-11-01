"""E. Text analytics (frequency based; model-based NLP lives in the nlp stage)."""
from __future__ import annotations

from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd

from ._common import text_messages


def top_words(df: pd.DataFrame, n: int = 20, sender: str | None = None) -> pd.DataFrame:
    tm = text_messages(df)
    if sender is not None:
        tm = tm[tm["sender"] == sender]
    c = Counter(chain.from_iterable(tm["tokens_content"]))
    return pd.DataFrame(c.most_common(n), columns=["word", "count"])


def ngrams(df: pd.DataFrame, n: int = 2, min_count: int = 3, top: int = 20,
           remove_stopwords: bool = True) -> pd.DataFrame:
    """n-grams within a message (never across message boundaries). With stopword removal the
    grams are built from content tokens, so they are 'content bigrams', not literal phrases."""
    tm = text_messages(df)
    col = "tokens_content" if remove_stopwords else "tokens"
    c: Counter = Counter()
    for toks in tm[col]:
        c.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    rows = [(" ".join(k), v) for k, v in c.most_common() if v >= min_count][:top]
    return pd.DataFrame(rows, columns=["ngram", "count"])


def frequent_messages(df: pd.DataFrame, min_count: int = 3, top: int = 20,
                      min_words: int = 2) -> pd.DataFrame:
    """Most repeated whole messages (normalised) - the 'phrases' people send verbatim."""
    tm = text_messages(df)
    tm = tm[tm["n_words"] >= min_words]
    vc = tm["text_clean"].str.strip().value_counts()
    vc = vc[vc >= min_count].head(top)
    return vc.rename_axis("phrase").reset_index(name="count")


def emoji_frequency(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    tm = text_messages(df)
    c = Counter(chain.from_iterable(tm["emojis"]))
    out = pd.DataFrame(c.most_common(top), columns=["emoji", "count"])
    total = sum(c.values())
    out["pct"] = 100 * out["count"] / total if total else 0.0
    return out


def emoji_by_sender(df: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    tm = text_messages(df)
    rows = []
    for who, g in tm.groupby("sender"):
        c = Counter(chain.from_iterable(g["emojis"]))
        rows += [{"sender": who, "emoji": e, "count": k} for e, k in c.most_common(top)]
    return pd.DataFrame(rows, columns=["sender", "emoji", "count"])


def domain_frequency(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    tm = text_messages(df)
    c = Counter(chain.from_iterable(tm["domains"]))
    return pd.DataFrame(c.most_common(top), columns=["domain", "count"])


def length_distribution(df: pd.DataFrame) -> dict:
    tm = text_messages(df)
    w = tm["n_words"]
    if w.empty:
        return {}
    bins = [0, 1, 3, 6, 11, 21, 51, np.inf]
    labels = ["1", "2-3", "4-6", "7-11", "12-21", "22-51", "52+"]
    hist = pd.cut(w, bins, labels=labels).value_counts().reindex(labels, fill_value=0)
    return {"histogram": hist.rename_axis("words").reset_index(name="messages"),
            "describe": w.describe(percentiles=[.5, .9, .99]).to_dict(),
            "note": "Distribution is heavy-tailed; prefer median/percentiles to the mean."}


def language_distribution(df: pd.DataFrame) -> pd.DataFrame:
    tm = text_messages(df)
    vc = tm["language"].value_counts()
    return pd.DataFrame({"language": vc.index, "messages": vc.to_numpy(),
                         "pct": 100 * vc.to_numpy() / vc.sum() if vc.sum() else 0.0})


def distinctive_vocabulary(df: pd.DataFrame, min_total: int = 5, top: int = 10) -> pd.DataFrame:
    """Participant-specific words via smoothed log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn 2008) and its z-score; each participant vs everyone else.
    Words with |z| >= 1.96 are 'distinctive' at roughly 5%, but treat as exploratory."""
    tm = text_messages(df)
    counts = {who: Counter(chain.from_iterable(g["tokens_content"]))
              for who, g in tm.groupby("sender")}
    total = Counter()
    for c in counts.values():
        total.update(c)
    vocab = [w for w, k in total.items() if k >= min_total]
    if not vocab or len(counts) < 2:
        return pd.DataFrame(columns=["sender", "word", "log_odds_z", "count"])
    a0 = 0.01 * sum(total[w] for w in vocab)
    prior = {w: 0.01 * total[w] * len(vocab) / max(len(vocab), 1) + 1e-3 for w in vocab}
    a0 = sum(prior.values())
    rows = []
    for who, c in counts.items():
        n1 = sum(c[w] for w in vocab)
        n2 = sum(total[w] - c[w] for w in vocab)
        if n1 == 0 or n2 == 0:
            continue
        res = []
        for w in vocab:
            y1, y2, a = c[w], total[w] - c[w], prior[w]
            d = (np.log((y1 + a) / (n1 + a0 - y1 - a))
                 - np.log((y2 + a) / (n2 + a0 - y2 - a)))
            var = 1 / (y1 + a) + 1 / (y2 + a)
            res.append((w, d / np.sqrt(var), y1))
        res.sort(key=lambda x: -x[1])
        rows += [{"sender": who, "word": w, "log_odds_z": float(z), "count": int(y)}
                 for w, z, y in res[:top] if y > 0]
    return pd.DataFrame(rows, columns=["sender", "word", "log_odds_z", "count"])
