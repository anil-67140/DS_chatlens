"""Topic discovery: six methods compared on coherence, diversity and cluster validity.

Methods: LDA, NMF, K-means on TF-IDF, LSA+K-means, sentence-embedding K-means, BERTopic.
These are UNSUPERVISED: there is no accuracy. Quality is judged by
  * NPMI coherence of each topic's top words (proxy for human interpretability),
  * topic diversity (non-redundancy),
  * silhouette / Davies-Bouldin / Calinski-Harabasz of the document assignments, measured in ONE
    shared space (TF-IDF -> SVD) so methods are comparable (BERTopic outliers excluded and reported),
  * and the top words + representative documents, for the human reading that no metric replaces.
The number of topics k is chosen by NPMI coherence on a sweep, never silently fixed.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from ..preprocessing import stopwords as sw
from ..preprocessing.normalize import normalize_text, tokenize
from . import evaluation as ev


@dataclass
class TopicResult:
    method: str
    k: int
    topics: list[list[str]]
    labels: np.ndarray                 # hard topic per document (-1 = outlier)
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)


def prepare_docs(texts: Sequence[str], min_tokens: int = 3) -> tuple[list[str], list[list[str]], list[int]]:
    """Normalise, tokenise and drop stopwords. Returns (clean_texts, token_lists, kept_indices);
    documents with fewer than ``min_tokens`` content tokens are excluded (and the caller can count
    them from ``kept_indices``)."""
    stop = sw.ALL | sw.CHAT_NOISE
    toks_all, keep = [], []
    for i, t in enumerate(texts):
        toks = [w for w in tokenize(normalize_text(t)) if w not in stop and len(w) > 2]
        if len(toks) >= min_tokens:
            toks_all.append(toks)
            keep.append(i)
    return [" ".join(t) for t in toks_all], toks_all, keep


def ctfidf_top_words(docs_tokens: list[list[str]], labels: np.ndarray, top_n: int = 10) -> list[list[str]]:
    """Class-based TF-IDF (as in BERTopic): words frequent in a cluster and rare elsewhere."""
    clusters = sorted(c for c in set(labels.tolist()) if c >= 0)
    vocab: dict[str, int] = {}
    for d in docs_tokens:
        for w in d:
            vocab.setdefault(w, len(vocab))
    tf = np.zeros((len(clusters), len(vocab)))
    cidx = {c: i for i, c in enumerate(clusters)}
    for d, lab in zip(docs_tokens, labels, strict=True):
        if lab >= 0:
            for w in d:
                tf[cidx[lab], vocab[w]] += 1
    avg = tf.sum() / max(len(clusters), 1)
    idf = np.log(1 + avg / np.maximum(tf.sum(0), 1))
    score = (tf / np.maximum(tf.sum(1, keepdims=True), 1)) * idf
    inv = {i: w for w, i in vocab.items()}
    return [[inv[j] for j in np.argsort(-score[i])[:top_n]] for i in range(len(clusters))]


def _top_words_from_components(comp: np.ndarray, vocab: np.ndarray, top_n: int) -> list[list[str]]:
    return [vocab[np.argsort(-row)[:top_n]].tolist() for row in comp]


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def fit_lda(docs, toks, k, seed=42, top_n=10, **_):
    cv = CountVectorizer(min_df=3, max_df=0.5, max_features=20000, token_pattern=r"\S+")
    X = cv.fit_transform(docs)

    def go():
        m = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=seed,
                                      max_iter=15, batch_size=512)
        return m, m.fit_transform(X)
    (m, theta), sec = _timed(go)
    return TopicResult("lda", k, _top_words_from_components(m.components_, cv.get_feature_names_out(), top_n),
                       theta.argmax(1), sec)


def fit_nmf(docs, toks, k, seed=42, top_n=10, **_):
    tv = TfidfVectorizer(min_df=3, max_df=0.5, max_features=20000, sublinear_tf=True, token_pattern=r"\S+")
    X = tv.fit_transform(docs)

    def go():
        m = NMF(n_components=k, init="nndsvd", random_state=seed, max_iter=400)
        return m, m.fit_transform(X)
    (m, W), sec = _timed(go)
    return TopicResult("nmf", k, _top_words_from_components(m.components_, tv.get_feature_names_out(), top_n),
                       W.argmax(1), sec)


def fit_kmeans_tfidf(docs, toks, k, seed=42, top_n=10, **_):
    tv = TfidfVectorizer(min_df=3, max_df=0.5, max_features=20000, sublinear_tf=True, token_pattern=r"\S+")
    X = tv.fit_transform(docs)
    km, sec = _timed(lambda: KMeans(k, n_init=5, random_state=seed).fit(X))
    words = _top_words_from_components(km.cluster_centers_, tv.get_feature_names_out(), top_n)
    return TopicResult("kmeans_tfidf", k, words, km.labels_, sec)


def fit_lsa_kmeans(docs, toks, k, seed=42, top_n=10, **_):
    tv = TfidfVectorizer(min_df=3, max_df=0.5, max_features=20000, sublinear_tf=True, token_pattern=r"\S+")
    X = tv.fit_transform(docs)

    def go():
        Z = normalize(TruncatedSVD(min(100, X.shape[1] - 1), random_state=seed).fit_transform(X))
        return KMeans(k, n_init=5, random_state=seed).fit(Z).labels_
    labels, sec = _timed(go)
    return TopicResult("lsa_kmeans", k, ctfidf_top_words(toks, labels, top_n), labels, sec)


def fit_embedding_kmeans(docs, toks, k, seed=42, top_n=10, encoder: Callable | None = None, texts=None, **_):
    """Sentence-embedding K-means. ``encoder(list[str]) -> ndarray`` (see :func:`make_st_encoder`)."""
    if encoder is None:
        raise ImportError("embedding_kmeans needs an encoder (sentence-transformers)")

    def go():
        E = normalize(np.asarray(encoder(texts if texts is not None else docs)))
        return KMeans(k, n_init=5, random_state=seed).fit(E).labels_
    labels, sec = _timed(go)
    return TopicResult("embedding_kmeans", k, ctfidf_top_words(toks, labels, top_n), labels, sec)


def fit_bertopic(docs, toks, k, seed=42, top_n=10, embeddings=None, texts=None, **_):
    """BERTopic (UMAP + HDBSCAN + c-TF-IDF) on precomputed embeddings. ``k`` is the *target* number
    of topics (reduce_topics); BERTopic decides the natural number and labels outliers -1."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    if embeddings is None:
        raise ImportError("bertopic needs precomputed embeddings")

    def go():
        um = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
        hd = HDBSCAN(min_cluster_size=max(10, len(docs) // 200), metric="euclidean",
                     cluster_selection_method="eom", prediction_data=False)
        bt = BERTopic(umap_model=um, hdbscan_model=hd, nr_topics=k, calculate_probabilities=False,
                      verbose=False)
        topics, _ = bt.fit_transform(docs, np.asarray(embeddings))
        return bt, np.asarray(topics)
    (bt, labels), sec = _timed(go)
    ids = sorted(t for t in set(labels.tolist()) if t >= 0)
    topics = [[w for w, _ in (bt.get_topic(t) or [])][:top_n] for t in ids]
    remap = {t: i for i, t in enumerate(ids)}
    labels = np.array([remap.get(t, -1) for t in labels])
    return TopicResult("bertopic", len(ids), topics, labels, sec, {"natural_topics": len(ids)})


METHODS: dict[str, Callable] = {"lda": fit_lda, "nmf": fit_nmf, "kmeans_tfidf": fit_kmeans_tfidf,
                                "lsa_kmeans": fit_lsa_kmeans}


def make_st_encoder(model_id: str = "paraphrase-multilingual-MiniLM-L12-v2", batch_size: int = 128):
    """Sentence-transformers encoder (multilingual: covers English/Hindi/Urdu script)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id)
    return lambda texts: model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                                      normalize_embeddings=True)


def evaluation_space(docs: list[str], seed: int = 42, dims: int = 50) -> np.ndarray:
    """Shared TF-IDF->SVD space in which every method's clustering is scored."""
    tv = TfidfVectorizer(min_df=3, max_df=0.5, max_features=20000, sublinear_tf=True, token_pattern=r"\S+")
    X = tv.fit_transform(docs)
    return normalize(TruncatedSVD(min(dims, X.shape[1] - 1), random_state=seed).fit_transform(X))


def score_result(res: TopicResult, toks: list[list[str]], space: np.ndarray, top_k: int = 10) -> dict:
    cm = ev.clustering_metrics(space, res.labels)
    coh = ev.npmi_coherence(res.topics, toks, top_k)
    return {"method": res.method, "k": res.k, "npmi_coherence": coh["mean"],
            "topic_diversity": ev.topic_diversity(res.topics, top_k),
            "silhouette": cm.get("silhouette"), "davies_bouldin": cm.get("davies_bouldin"),
            "calinski_harabasz": cm.get("calinski_harabasz"),
            "noise_fraction": cm["noise_fraction"], "seconds": round(res.seconds, 2),
            "per_topic_npmi": coh["per_topic"]}


def sweep_k(method: str, docs, toks, ks=(5, 10, 15, 20), seed=42, space=None, **kw) -> list[dict]:
    space = space if space is not None else evaluation_space(docs, seed)
    fn = METHODS.get(method) or {"embedding_kmeans": fit_embedding_kmeans, "bertopic": fit_bertopic}[method]
    out = []
    for k in ks:
        r = fn(docs, toks, k, seed=seed, **kw)
        s = score_result(r, toks, space)
        s["result"] = r
        out.append(s)
    return out


def best_k(rows: list[dict], min_diversity: float = 0.5, tolerance: float = 0.02) -> dict:
    """Pick k: among non-redundant settings (diversity >= ``min_diversity``), take those whose NPMI
    coherence is within ``tolerance`` of the best, then choose the highest silhouette.

    Why not coherence alone? Coherence does not penalise a model that silently drops a whole theme:
    a k that merges/omits a topic can score marginally higher. Near-ties are therefore decided by
    cluster separation. This is a heuristic; always read the topics."""
    ok = [r for r in rows if r["topic_diversity"] >= min_diversity] or rows
    top = max(r["npmi_coherence"] for r in ok)
    near = [r for r in ok if r["npmi_coherence"] >= top - tolerance]
    return max(near, key=lambda r: (r["silhouette"] if r["silhouette"] is not None else -1.0, r["npmi_coherence"]))


def representative_docs(res: TopicResult, texts: Sequence[str], space: np.ndarray, n: int = 3) -> dict:
    """The documents closest to each topic's centroid in the shared space (for human review)."""
    out = {}
    for t in range(len(res.topics)):
        idx = np.where(res.labels == t)[0]
        if not len(idx):
            continue
        c = space[idx].mean(0)
        order = idx[np.argsort(-(space[idx] @ c))[:n]]
        out[t] = [texts[i] for i in order]
    return out
