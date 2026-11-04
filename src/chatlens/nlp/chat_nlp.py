"""Applying trained/zero-shot models to a chat, with language routing and honest coverage.

A model trained on English must not silently score Urdu script, so each message is routed by its
detected language to a model registered for that language; messages with no suitable model are
reported as ``unscored`` (and counted) instead of being forced into a label."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import stats
from .classifiers import LexiconClassifier

POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


class LabelModel:
    """Adapter giving any fitted pipeline/classifier a common ``predict_label`` interface."""

    def __init__(self, name: str, model, classes: list[str] | None = None, has_proba: bool = True):
        self.name, self.model = name, model
        self.classes_ = list(classes or getattr(model, "classes_", []))
        self.has_proba = has_proba and hasattr(model, "predict_proba")

    def predict_label(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
        if self.has_proba:
            try:
                p = np.asarray(self.model.predict_proba(texts))
                return np.array(self.classes_)[p.argmax(1)], p.max(1)
            except NotImplementedError:
                pass
        return np.asarray(self.model.predict(texts)), None


def vader_model() -> LabelModel:
    m = LexiconClassifier("vader")
    m.classes_ = ["negative", "neutral", "positive"]
    return LabelModel("vader(zero-shot, English only)", m, m.classes_, has_proba=False)


class LanguageRouter:
    """Route messages to a per-language model. ``models`` maps language code -> LabelModel."""

    def __init__(self, models: dict[str, LabelModel], task: str = "sentiment"):
        self.models, self.task = models, task

    def score(self, df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
        sub = df[df["is_text"] & (~df["is_system"])]
        out = pd.DataFrame({"msg_id": sub["msg_id"].to_numpy(), "label": "unscored",
                            "confidence": np.nan, "method": ""})
        for lang, g in sub.groupby("language"):
            model = self.models.get(lang)
            if model is None:
                continue
            labels, conf = model.predict_label(g[text_col].tolist())
            sel = out["msg_id"].isin(g["msg_id"])
            out.loc[sel, "label"] = labels
            out.loc[sel, "method"] = model.name
            if conf is not None:
                out.loc[sel, "confidence"] = conf
        return out


def summarise_sentiment(df: pd.DataFrame, scored: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Coverage, per-participant and per-month aggregates, and a participant difference test."""
    m = df.merge(scored[["msg_id", "label", "confidence"]], on="msg_id", how="inner")
    total = len(m)
    sc = m[m["label"] != "unscored"].copy()
    res: dict = {"n_text_messages": total, "n_scored": len(sc),
                 "coverage": len(sc) / total if total else 0.0,
                 "coverage_by_language": {lang: float((g["label"] != "unscored").mean())
                                          for lang, g in m.groupby("language")},
                 "label_counts": sc["label"].value_counts().to_dict()}
    if sc.empty:
        return res
    sc["polarity"] = sc["label"].map(POLARITY)
    by = sc.groupby("sender")
    present = [c for c in ("positive", "neutral", "negative") if c in set(sc["label"])]
    cols = {f"pct_{c}": by["label"].apply(lambda s, c=c: 100 * (s == c).mean()) for c in present}
    res["by_participant"] = pd.DataFrame({"scored": by.size(), "mean_polarity": by["polarity"].mean(), **cols}
                                         ).sort_values("scored", ascending=False)
    valid = sc[sc["timestamp"].notna()]
    if not valid.empty:
        mo = valid.assign(month=valid["timestamp"].dt.to_period("M").dt.to_timestamp()).groupby("month")["polarity"]
        res["monthly_polarity"] = pd.DataFrame({"mean_polarity": mo.mean(), "n": mo.size()})
    big = sc["sender"].value_counts()
    big = big[big >= 30].index
    test = stats.chi_square_independence(sc.loc[sc["sender"].isin(big), "sender"],
                                         sc.loc[sc["sender"].isin(big), "label"], alpha)
    res["participant_difference_test"] = test.to_dict() if test else None
    if sc["confidence"].notna().any():
        res["mean_confidence"] = float(sc["confidence"].mean())
    res["caveat"] = ("Sentiment labels are model predictions with the error rates in the benchmark report; "
                     "per-person differences smaller than the model's error are not meaningful.")
    return res


Factory = Callable[[], LanguageRouter]
