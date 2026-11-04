"""Uniform interface for text classifiers used in the sentiment and emotion benchmarks.

Every method exposes ``fit(train, val)`` / ``predict(texts)`` / ``predict_proba(texts)`` with a
common class list. Methods that cannot produce probabilities (the lexicon baselines) say so with
``has_proba = False`` and are then excluded from AUC/calibration metrics rather than given
invented probabilities.

Families
  baseline     majority class (the floor every method must beat)
  lexicon      VADER, TextBlob: zero-shot, English only; ``_tuned`` variants learn the neutral band
  tfidf        TF-IDF (word / char / both) + LR / SVM / NB / SGD / RF; hyper-parameters are
               chosen on the validation split only
  pretrained   Hugging Face models used zero-shot (needs the transformers package + GPU/CPU)
  finetuned    Hugging Face encoders fine-tuned on the training split (Colab GPU)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ..preprocessing.normalize import normalize_for_classification, tokenize


class BaseClassifier(ABC):
    name: str = "base"
    family: str = "base"
    has_proba: bool = True
    zero_shot: bool = False          # True if the model is never trained on our training split

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self.fit_seconds: float = 0.0
        self.best_params_: dict = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> BaseClassifier:
        t0 = time.perf_counter()
        self.classes_ = sorted(train["label"].unique())
        self._fit(train["text"].tolist(), train["label"].tolist(),
                  None if val is None else val["text"].tolist(),
                  None if val is None else val["label"].tolist())
        self.fit_seconds = time.perf_counter() - t0
        return self

    @abstractmethod
    def _fit(self, X, y, Xv, yv) -> None: ...

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(self.classes_)[self.predict_proba(texts).argmax(1)]


def _macro_f1(y, p, classes) -> float:
    return f1_score(y, p, labels=classes, average="macro", zero_division=0)


# --------------------------------------------------------------------------------- baseline
class MajorityBaseline(BaseClassifier):
    name, family = "majority_baseline", "baseline"

    def _fit(self, X, y, Xv, yv):
        self.m = DummyClassifier(strategy="prior").fit(np.zeros((len(y), 1)), y)

    def predict_proba(self, texts):
        return self.m.predict_proba(np.zeros((len(texts), 1)))


# ---------------------------------------------------------------------------------- lexicons
class LexiconClassifier(BaseClassifier):
    """Polarity score in [-1, 1] -> label. ``tune=False``: the published +/-0.05 neutral band
    (pure zero-shot). ``tune=True``: the band is chosen on the training split to maximise
    macro-F1 (so the lexicon gets the same chance to adapt as the supervised models)."""

    family, has_proba = "lexicon", False

    def __init__(self, kind: str, tune: bool = False):
        super().__init__()
        self.kind, self.tune = kind, tune
        self.name = f"{kind}{'_tuned' if tune else ''}"
        self.zero_shot = not tune
        self.band = 0.05
        self._scorer: Callable[[str], float] | None = None

    def _scorer_fn(self) -> Callable[[str], float]:
        if self._scorer is None:
            if self.kind == "vader":
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                an = SentimentIntensityAnalyzer()
                self._scorer = lambda t: an.polarity_scores(t)["compound"]
            elif self.kind == "textblob":
                from textblob import TextBlob
                self._scorer = lambda t: float(TextBlob(t).sentiment.polarity)
            else:
                raise ValueError(self.kind)
        return self._scorer

    def scores(self, texts):
        f = self._scorer_fn()
        return np.array([f(t) for t in texts])

    def _labels_from(self, s, band):
        out = np.where(s > band, "positive", np.where(s < -band, "negative", "neutral"))
        if "neutral" not in self.classes_:
            out = np.where(s >= 0, "positive", "negative")
        return out

    def _fit(self, X, y, Xv, yv):
        if not self.tune:
            return
        s, y = self.scores(X), np.asarray(y)
        grid = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        best = max(grid, key=lambda b: _macro_f1(y, self._labels_from(s, b), self.classes_))
        self.band = best
        self.best_params_ = {"neutral_band": best}

    def predict_proba(self, texts):
        raise NotImplementedError("lexicon methods provide no probabilities")

    def predict(self, texts):
        return self._labels_from(self.scores(texts), self.band)


# ------------------------------------------------------------------------------------ tf-idf
def _word_vec():
    return TfidfVectorizer(preprocessor=normalize_for_classification, ngram_range=(1, 2),
                           min_df=2, sublinear_tf=True, max_features=200_000, dtype=np.float32,
                           tokenizer=tokenize, token_pattern=None, lowercase=False)


def _char_vec():
    return TfidfVectorizer(preprocessor=normalize_for_classification, analyzer="char_wb",
                           ngram_range=(2, 4), min_df=3, sublinear_tf=True, max_features=150_000,
                           dtype=np.float32)


def _both_vec():
    return FeatureUnion([("w", _word_vec()), ("c", _char_vec())])


class SklearnTextClassifier(BaseClassifier):
    family = "tfidf"

    def __init__(self, name: str, vec_factory, clf_factory, grid: list[dict] | None = None,
                 seed: int = 42):
        super().__init__()
        self.name, self._vf, self._cf, self.grid, self.seed = name, vec_factory, clf_factory, \
            (grid or [{}]), seed

    def _fit(self, X, y, Xv, yv):
        # Vectoriser is fitted ONCE (on the training split only); only the classifier's
        # hyper-parameters are searched, using the validation split.
        y = np.asarray(y)          # older scikit-learn (<1.5) mis-handles list labels in calibration
        yv = None if yv is None else np.asarray(yv)
        vec = self._vf().fit(X)
        Xt = vec.transform(X)
        Xvt = vec.transform(Xv) if Xv is not None else None
        best, best_params, best_score = None, {}, -1.0
        for params in self.grid:
            clf = self._cf(**params).fit(Xt, y)
            if Xvt is None or len(self.grid) == 1:
                best, best_params, best_score = clf, params, 0.0
                break
            sc = _macro_f1(yv, clf.predict(Xvt), self.classes_)       # validation only, never test
            if sc > best_score:
                best, best_params, best_score = clf, params, sc
        self.model = Pipeline([("vec", vec), ("clf", best)])
        self.best_params_ = best_params
        self.val_macro_f1 = best_score if best_score > 0 else None

    def predict_proba(self, texts):
        p = self.model.predict_proba(list(texts))
        order = [list(self.model.classes_).index(c) for c in self.classes_]
        return p[:, order]


def tfidf_methods(seed: int = 42) -> list[SklearnTextClassifier]:
    lr = lambda **k: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed,  # noqa: E731
                                        **k)
    C = [{"C": c} for c in (0.5, 2.0, 8.0)]
    return [
        SklearnTextClassifier("tfidf_word_logreg", _word_vec, lr, C, seed),
        SklearnTextClassifier("tfidf_char_logreg", _char_vec, lr, C, seed),
        SklearnTextClassifier("tfidf_word+char_logreg", _both_vec, lr, C, seed),
        SklearnTextClassifier(
            "tfidf_word+char_linearsvc_calibrated", _both_vec,
            lambda C=1.0: CalibratedClassifierCV(
                LinearSVC(C=C, class_weight="balanced", random_state=seed), cv=3),
            [{"C": c} for c in (0.1, 0.5, 2.0)], seed),
        SklearnTextClassifier("tfidf_word_complementnb", _word_vec,
                              lambda alpha=0.5: ComplementNB(alpha=alpha),
                              [{"alpha": a} for a in (0.1, 0.5, 1.0)], seed),
        SklearnTextClassifier("tfidf_word_multinomialnb", _word_vec,
                              lambda alpha=0.5: MultinomialNB(alpha=alpha),
                              [{"alpha": a} for a in (0.1, 0.5, 1.0)], seed),
        SklearnTextClassifier(
            "tfidf_word+char_sgd_modhuber", _both_vec,
            lambda alpha=1e-5: SGDClassifier(loss="modified_huber", alpha=alpha,
                                             class_weight="balanced", random_state=seed),
            [{"alpha": a} for a in (1e-6, 1e-5, 1e-4)], seed),
        SklearnTextClassifier(
            "tfidf_word_randomforest",
            lambda: TfidfVectorizer(preprocessor=normalize_for_classification, min_df=3,
                                    sublinear_tf=True, max_features=30_000,
                                    tokenizer=tokenize, token_pattern=None, lowercase=False),
            lambda **k: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                               class_weight="balanced_subsample",
                                               random_state=seed), None, seed),
    ]


# ------------------------------------------------------------------ Hugging Face (lazy import)
def _hf_pipeline(model_id: str, device: int | None = None, batch_size: int = 64):
    import torch
    from transformers import pipeline
    dev = device if device is not None else (0 if torch.cuda.is_available() else -1)
    return pipeline("text-classification", model=model_id, tokenizer=model_id, top_k=None,
                    truncation=True, max_length=128, device=dev, batch_size=batch_size)


class HFPretrainedClassifier(BaseClassifier):
    """Zero-shot use of a released classification model.

    ``label_map`` maps the model's labels onto ours (case-insensitive). Dataset classes the model
    cannot output get probability 0. If the dataset has a 'neutral' class but the model does not,
    a margin threshold is learned on the training split: |p_pos - p_neg| < tau -> neutral."""

    family, zero_shot = "pretrained", True

    def __init__(self, name: str, model_id: str, label_map: dict[str, str],
                 pipeline_factory: Callable | None = None, batch_size: int = 64):
        super().__init__()
        self.name, self.model_id = name, model_id
        self.label_map = {k.lower(): v for k, v in label_map.items()}
        self._factory = pipeline_factory or (lambda: _hf_pipeline(model_id, batch_size=batch_size))
        self._pipe = None
        self.tau: float | None = None

    def _raw_proba(self, texts) -> np.ndarray:
        if self._pipe is None:
            self._pipe = self._factory()
        outs = self._pipe(list(texts))
        P = np.zeros((len(outs), len(self.classes_)))
        for i, scores in enumerate(outs):
            for s in scores:
                tgt = self.label_map.get(str(s["label"]).lower())
                if tgt in self.classes_:
                    P[i, self.classes_.index(tgt)] += float(s["score"])
        return P

    def _fit(self, X, y, Xv, yv):
        covered = set(self.label_map.values())
        missing = set(self.classes_) - covered
        if missing == {"neutral"} and {"positive", "negative"} <= covered:
            P = self._raw_proba(X[:3000])
            ip, ineg = self.classes_.index("positive"), self.classes_.index("negative")
            margin = np.abs(P[:, ip] - P[:, ineg])
            yy = np.asarray(y[:3000])
            best, best_f = 0.0, -1.0
            for tau in np.linspace(0, 1, 21):
                lab = np.where(margin < tau, "neutral",
                               np.where(P[:, ip] >= P[:, ineg], "positive", "negative"))
                f = _macro_f1(yy, lab, self.classes_)
                if f > best_f:
                    best, best_f = tau, f
            self.tau = float(best)
            self.best_params_ = {"neutral_margin_tau": self.tau}
        elif missing:
            self.best_params_ = {"classes_never_predicted": sorted(missing)}

    def predict_proba(self, texts):
        P = self._raw_proba(texts)
        if self.tau is not None and "neutral" in self.classes_:
            ip, ineg, inu = (self.classes_.index(c) for c in ("positive", "negative", "neutral"))
            m = np.abs(P[:, ip] - P[:, ineg])
            neutral = m < self.tau
            P = P.copy()
            P[neutral, inu] = 1.0 - m[neutral] / max(self.tau, 1e-9) * 0.5
            P[neutral, ip] *= 0.5
            P[neutral, ineg] *= 0.5
        s = P.sum(1, keepdims=True)
        return np.divide(P, s, out=np.full_like(P, 1 / P.shape[1]), where=s > 0)


class HFFineTunedClassifier(BaseClassifier):
    """Fine-tune an encoder with a plain PyTorch loop (no Trainer API dependency).

    Resumable: weights are saved to ``ckpt_dir`` after every epoch together with the epoch
    counter, so a Colab disconnect costs at most one epoch."""

    family = "finetuned"

    def __init__(self, name: str, model_id: str, ckpt_dir: str, epochs: int = 3, lr: float = 2e-5,
                 batch_size: int = 32, max_length: int = 128, seed: int = 42, patience: int = 1,
                 max_train: int | None = None):
        super().__init__()
        self.name, self.model_id, self.ckpt_dir = name, model_id, ckpt_dir
        self.epochs, self.lr, self.bs, self.max_len = epochs, lr, batch_size, max_length
        self.seed, self.patience, self.max_train = seed, patience, max_train
        self.history: list[dict] = []

    def _fit(self, X, y, Xv, yv):
        import json
        import os

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.max_train and len(X) > self.max_train:
            sel = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X, y = [X[i] for i in sel], [y[i] for i in sel]
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        c2i = {c: i for i, c in enumerate(self.classes_)}
        state_path = os.path.join(self.ckpt_dir, "state.json")
        start_epoch, best_f1, bad = 0, -1.0, 0
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=len(self.classes_),
            id2label=dict(enumerate(self.classes_)), label2id=c2i,
            ignore_mismatched_sizes=True)
        weights = os.path.join(self.ckpt_dir, "last.pt")
        if os.path.exists(state_path) and os.path.exists(weights):
            st = json.load(open(state_path))
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            start_epoch, best_f1, bad, self.history = (st["epoch"], st["best_f1"], st["bad"],
                                                       st["history"])
        self.model.to(dev)
        yi = np.array([c2i[c] for c in y])
        steps = max(1, (len(X) + self.bs - 1) // self.bs) * self.epochs
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=dev == "cuda")
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            perm = np.random.default_rng(self.seed + epoch).permutation(len(X))
            losses = []
            for i in range(0, len(X), self.bs):
                b = perm[i:i + self.bs]
                enc = self.tok([X[j] for j in b], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(dev)
                labels = torch.tensor(yi[b]).to(dev)
                with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                    loss = self.model(**enc, labels=labels).loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                losses.append(float(loss))
            rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
            if Xv is not None:
                pv = self._proba_raw(Xv, dev)
                rec["val_macro_f1"] = _macro_f1(yv, np.array(self.classes_)[pv.argmax(1)],
                                                self.classes_)
                if rec["val_macro_f1"] > best_f1:
                    best_f1, bad = rec["val_macro_f1"], 0
                    torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
                else:
                    bad += 1
            self.history.append(rec)
            torch.save(self.model.state_dict(), weights)
            with open(state_path, "w") as fh:
                json.dump({"epoch": epoch + 1, "best_f1": best_f1, "bad": bad,
                           "history": self.history}, fh)
            if bad > self.patience:
                break
        best = os.path.join(self.ckpt_dir, "best.pt")
        if os.path.exists(best):
            self.model.load_state_dict(torch.load(best, map_location=dev))
        self.model.eval()
        self.best_params_ = {"epochs_run": len(self.history), "lr": self.lr}

    def _proba_raw(self, texts, dev=None):
        import torch
        dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.bs * 2):
                enc = self.tok(list(texts[i:i + self.bs * 2]), padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(dev)
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes_)))

    def predict_proba(self, texts):
        return self._proba_raw(list(texts))


# -------------------------------------------------------------------------------- registries
SENTIMENT_LABELS = {"negative": "negative", "neutral": "neutral", "positive": "positive",
                    "label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def sentiment_registry(seed: int = 42, ckpt_root: str = "/tmp/ckpt", include_hf: bool = True,
                       finetune_epochs: int = 3, finetune_max_train: int | None = None):
    """All sentiment methods compared in the project, in a stable order."""
    ms: list[BaseClassifier] = [MajorityBaseline(),
                                LexiconClassifier("vader"), LexiconClassifier("vader", tune=True),
                                LexiconClassifier("textblob"),
                                LexiconClassifier("textblob", tune=True)]
    ms += tfidf_methods(seed)
    if include_hf:
        ms += [
            HFPretrainedClassifier("hf_xlmr_twitter_sentiment",
                                   "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                                   SENTIMENT_LABELS),
            HFPretrainedClassifier("hf_distilbert_multilingual_sentiment",
                                   "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                                   SENTIMENT_LABELS),
            HFPretrainedClassifier("hf_distilbert_sst2_english",
                                   "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                   {"POSITIVE": "positive", "NEGATIVE": "negative"}),
            HFFineTunedClassifier("ft_xlm_roberta_base", "xlm-roberta-base",
                                  f"{ckpt_root}/ft_xlmr", epochs=finetune_epochs, seed=seed,
                                  max_train=finetune_max_train),
            HFFineTunedClassifier("ft_multilingual_bert", "bert-base-multilingual-cased",
                                  f"{ckpt_root}/ft_mbert", epochs=finetune_epochs, seed=seed,
                                  max_train=finetune_max_train),
            HFFineTunedClassifier("ft_distilbert_multilingual",
                                  "distilbert/distilbert-base-multilingual-cased",
                                  f"{ckpt_root}/ft_distilmbert", epochs=finetune_epochs,
                                  seed=seed, max_train=finetune_max_train),
        ]
    return ms


EMOTION_LABELS = {"joy": "joy", "love": "love", "sadness": "sadness", "anger": "anger",
                  "fear": "fear", "surprise": "surprise"}


def emotion_registry(seed: int = 42, ckpt_root: str = "/tmp/ckpt", include_hf: bool = True,
                     finetune_epochs: int = 3, finetune_max_train: int | None = None):
    """Emotion methods (English, 6 classes). No lexicon sentiment tools: they output polarity,
    not emotions, so comparing them here would be comparing different tasks."""
    ms: list[BaseClassifier] = [MajorityBaseline()] + tfidf_methods(seed)
    if include_hf:
        ms += [
            HFPretrainedClassifier("hf_distilroberta_emotion",
                                   "j-hartmann/emotion-english-distilroberta-base",
                                   {"joy": "joy", "sadness": "sadness", "anger": "anger",
                                    "fear": "fear", "surprise": "surprise"}),
            HFFineTunedClassifier("ft_distilroberta", "distilbert/distilroberta-base",
                                  f"{ckpt_root}/ft_emo_droberta", epochs=finetune_epochs,
                                  seed=seed, max_train=finetune_max_train),
            HFFineTunedClassifier("ft_bert_base_uncased", "bert-base-uncased",
                                  f"{ckpt_root}/ft_emo_bert", epochs=finetune_epochs, seed=seed,
                                  max_train=finetune_max_train),
        ]
    return ms
