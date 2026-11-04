"""Keyword / key-phrase extraction: six methods compared.

TF-IDF, RAKE (own implementation), YAKE, TextRank (own), embedding-MMR (KeyBERT-style) and a raw
frequency baseline. Without gold keyphrases there is no accuracy to report. The comparison uses
  * agreement: pairwise Jaccard of the top-k lists,
  * stability: Jaccard of top-k between two bootstrap resamples of the documents (a method whose
    keywords change when you resample 63% of the data is mostly noise),
  * runtime.
Agreement says methods are alike, not that they are right; read the lists.
"""
from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..preprocessing import stopwords as sw
from ..preprocessing.normalize import normalize_text, tokenize

STOP = sw.ALL | sw.CHAT_NOISE


def _tok(text: str) -> list[str]:
    """Unicode-aware content tokens (keeps Devanagari/Arabic-script words intact; sklearn's default
    pattern would split them at vowel signs)."""
    return [w for w in tokenize(text) if len(w) >= 3 and w not in STOP]


def _clean(texts: Sequence[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


def frequency_keywords(texts: Sequence[str], top_n: int = 20, **_) -> list[tuple[str, float]]:
    c = Counter(w for t in _clean(texts) for w in tokenize(t) if w not in STOP and len(w) > 2)
    return [(w, float(n)) for w, n in c.most_common(top_n)]


def tfidf_keywords(texts: Sequence[str], top_n: int = 20, ngram=(1, 2), **_):
    cl = _clean(texts)
    tv = TfidfVectorizer(ngram_range=ngram, min_df=2, max_df=0.7, sublinear_tf=True, tokenizer=_tok,
                         token_pattern=None, lowercase=False)
    X = tv.fit_transform(cl)
    score = np.asarray(X.sum(0)).ravel()
    names = tv.get_feature_names_out()
    return [(names[i], float(score[i])) for i in np.argsort(-score)[:top_n]]


def rake_keywords(texts: Sequence[str], top_n: int = 20, max_words: int = 3, **_):
    """RAKE (Rose et al. 2010): candidates are runs of non-stopwords; word score = degree/frequency."""
    freq: Counter = Counter()
    deg: Counter = Counter()
    phrases: Counter = Counter()
    for t in _clean(texts):
        for chunk in re.split(r"[.,;:!?\n()\[\]\"]+", t):
            cur: list[str] = []
            for w in tokenize(chunk) + [None]:
                if w is None or w in STOP or len(w) < 3:
                    # A run longer than max_words is not a phrase: fall back to its single words
                    # rather than discarding it (so unpunctuated chat text still yields keywords).
                    cands = [cur] if 0 < len(cur) <= max_words else [[x] for x in cur]
                    for cand in cands:
                        phrases[" ".join(cand)] += 1
                        for x in cand:
                            freq[x] += 1
                            deg[x] += len(cand)
                    cur = []
                else:
                    cur.append(w)
    scored = {p: sum(deg[w] / freq[w] for w in p.split()) for p in phrases if phrases[p] >= 2}
    return sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]


def yake_keywords(texts: Sequence[str], top_n: int = 20, lang: str = "en", **_):
    """YAKE on the concatenated corpus (statistical, unsupervised). Lower YAKE score = better; we
    return 1/score so that higher is better like the other methods."""
    import yake
    text = ". ".join(texts)[:2_000_000]
    ex = yake.KeywordExtractor(lan=lang, n=2, top=top_n * 3, dedupLim=0.7)
    kws = [(k, 1.0 / (s + 1e-9)) for k, s in ex.extract_keywords(text)]
    kws = [(k.lower(), s) for k, s in kws if not all(w in STOP for w in k.lower().split())]
    return kws[:top_n]


def textrank_keywords(texts: Sequence[str], top_n: int = 20, window: int = 4, d: float = 0.85,
                      iters: int = 50, **_):
    """TextRank (Mihalcea & Tarau 2004) on the corpus-level word co-occurrence graph."""
    edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in _clean(texts):
        toks = [w for w in tokenize(t) if w not in STOP and len(w) > 2]
        for i, w in enumerate(toks):
            for v in toks[i + 1:i + window]:
                if v != w:
                    edges[w][v] += 1
                    edges[v][w] += 1
    nodes = sorted(edges)
    if not nodes:
        return []
    idx = {w: i for i, w in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n))
    for w, nb in edges.items():
        for v, c in nb.items():
            M[idx[v], idx[w]] = c
    colsum = M.sum(0)
    M = np.divide(M, colsum, out=np.zeros_like(M), where=colsum > 0)
    r = np.full(n, 1 / n)
    for _ in range(iters):
        r_new = (1 - d) / n + d * M @ r
        if np.abs(r_new - r).sum() < 1e-9:
            r = r_new
            break
        r = r_new
    return [(nodes[i], float(r[i])) for i in np.argsort(-r)[:top_n]]


def embedding_mmr_keywords(texts: Sequence[str], top_n: int = 20, encoder: Callable | None = None,
                           diversity: float = 0.5, n_candidates: int = 200, **_):
    """KeyBERT-style: candidates are n-grams; rank by cosine to the corpus embedding with Maximal
    Marginal Relevance to avoid near-duplicates. ``encoder(list[str]) -> ndarray``."""
    if encoder is None:
        raise ImportError("embedding_mmr needs an encoder (sentence-transformers)")
    cl = _clean(texts)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, tokenizer=_tok, token_pattern=None, lowercase=False,
                         max_features=n_candidates * 5)
    X = cv.fit_transform(cl)
    names = cv.get_feature_names_out()
    top = np.argsort(-np.asarray(X.sum(0)).ravel())[:n_candidates]
    cands = [names[i] for i in top]
    if not cands:
        return []
    E = np.asarray(encoder(cands), float)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    doc = np.asarray(encoder([" ".join(cl)[:20000]]), float)[0]
    doc /= np.linalg.norm(doc) + 1e-12
    rel = E @ doc
    chosen = [int(np.argmax(rel))]
    while len(chosen) < min(top_n, len(cands)):
        rest = [i for i in range(len(cands)) if i not in chosen]
        red = (E[rest] @ E[chosen].T).max(1)
        mmr = (1 - diversity) * rel[rest] - diversity * red
        chosen.append(rest[int(np.argmax(mmr))])
    return [(cands[i], float(rel[i])) for i in chosen]


METHODS: dict[str, Callable] = {"frequency": frequency_keywords, "tfidf": tfidf_keywords,
                                "rake": rake_keywords, "yake": yake_keywords,
                                "textrank": textrank_keywords,
                                "embedding_mmr": embedding_mmr_keywords}


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def run_all(texts: Sequence[str], top_n: int = 20, methods: Sequence[str] | None = None,
            encoder: Callable | None = None, stability_runs: int = 3, seed: int = 42) -> dict:
    """Run every method, then agreement + bootstrap stability. Methods that raise (e.g. no encoder)
    are reported in ``errors`` and skipped."""
    names = list(methods or METHODS)
    out: dict = {"keywords": {}, "seconds": {}, "errors": {}}
    for m in names:
        try:
            t0 = time.perf_counter()
            out["keywords"][m] = METHODS[m](texts, top_n=top_n, encoder=encoder)
            out["seconds"][m] = round(time.perf_counter() - t0, 3)
        except Exception as exc:        # noqa: BLE001
            out["errors"][m] = f"{type(exc).__name__}: {exc}"
    ok = list(out["keywords"])
    lists = {m: [k for k, _ in out["keywords"][m]] for m in ok}
    out["agreement_jaccard"] = {f"{a}|{b}": jaccard(lists[a], lists[b]) for a, b in combinations(ok, 2)}
    rng = np.random.default_rng(seed)
    n = len(texts)
    stab: dict[str, list[float]] = {m: [] for m in ok}
    for _ in range(stability_runs):
        s1 = [texts[i] for i in rng.integers(0, n, n)]
        s2 = [texts[i] for i in rng.integers(0, n, n)]
        for m in ok:
            try:
                a = [k for k, _ in METHODS[m](s1, top_n=top_n, encoder=encoder)]
                b = [k for k, _ in METHODS[m](s2, top_n=top_n, encoder=encoder)]
                stab[m].append(jaccard(a, b))
            except Exception:        # noqa: BLE001
                pass
    out["stability_jaccard"] = {m: float(np.mean(v)) if v else None for m, v in stab.items()}
    return out
