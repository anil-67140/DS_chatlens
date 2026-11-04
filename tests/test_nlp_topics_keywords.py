"""Topic/keyword methods are checked on SYNTHETIC corpora with known planted structure."""
import numpy as np
import pytest

from chatlens.nlp import keywords as kw
from chatlens.nlp import topics as tp

THEMES = {
    "cricket": "cricket match wicket bowler batsman stadium innings captain",
    "cooking": "recipe kitchen spices chicken biryani oven flour simmer",
    "travel": "airport flight hotel passport luggage tourist beach journey",
    "tech": "laptop software python server database network keyboard compiler",
}


def corpus(n_per=80, seed=0):
    rng = np.random.default_rng(seed)
    texts, truth = [], []
    for ti, (_, vocab) in enumerate(THEMES.items()):
        words = vocab.split()
        for _ in range(n_per):
            texts.append(" ".join(rng.choice(words, 6)) + " " + rng.choice(["really", "today", "nicely"]))
            truth.append(ti)
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], np.array(truth)[order]


@pytest.fixture(scope="module")
def prepared():
    texts, truth = corpus()
    docs, toks, keep = tp.prepare_docs(texts)
    return texts, truth, docs, toks, keep


def purity(labels, truth):
    return sum(np.bincount(truth[labels == c]).max() for c in set(labels.tolist()) if c >= 0) / len(truth)


def test_prepare_docs_drops_short_and_stopword_only():
    docs, toks, keep = tp.prepare_docs(["the and of", "hi", "cricket match wicket bowler", "ok ok ok"])
    assert keep == [2] and toks[0][0] == "cricket"


@pytest.mark.parametrize("method", ["lda", "nmf", "kmeans_tfidf", "lsa_kmeans"])
def test_each_method_recovers_planted_topics(prepared, method):
    texts, truth, docs, toks, keep = prepared
    res = tp.METHODS[method](docs, toks, 4, seed=1)
    assert len(res.topics) == 4 and len(res.labels) == len(docs)
    assert purity(res.labels, truth[keep]) > 0.9, method
    planted = {w for v in THEMES.values() for w in v.split()}
    assert all(w in planted | {"really", "today", "nicely"} for t in res.topics for w in t[:5])


def test_scoring_and_k_selection_prefers_true_k(prepared):
    _, _, docs, toks, _ = prepared
    space = tp.evaluation_space(docs)
    rows = tp.sweep_k("nmf", docs, toks, ks=(2, 4, 8), space=space)
    best = tp.best_k(rows)
    assert best["k"] == 4 and best["npmi_coherence"] > 0.3
    r4 = next(r for r in rows if r["k"] == 4)
    assert r4["silhouette"] > 0.3 and r4["topic_diversity"] > 0.8   # 3 shared filler words lower it
    assert next(r for r in rows if r["k"] == 8)["topic_diversity"] < r4["topic_diversity"]


def test_embedding_kmeans_with_injected_encoder(prepared):
    texts, truth, docs, toks, keep = prepared
    theme_of = {w: i for i, v in enumerate(THEMES.values()) for w in v.split()}

    def fake_encoder(ts):   # a deterministic stand-in for sentence-transformers
        out = np.zeros((len(ts), 4))
        for i, t in enumerate(ts):
            for w in t.split():
                if w in theme_of:
                    out[i, theme_of[w]] += 1
        return out + 1e-3
    res = tp.fit_embedding_kmeans(docs, toks, 4, encoder=fake_encoder, texts=docs)
    assert purity(res.labels, truth[keep]) > 0.95
    with pytest.raises(ImportError):
        tp.fit_embedding_kmeans(docs, toks, 4)


def test_ctfidf_top_words_and_representatives(prepared):
    texts, truth, docs, toks, keep = prepared
    tw = tp.ctfidf_top_words(toks, truth[keep], 5)
    for ti, words in enumerate(tw):
        assert set(words) <= set(list(THEMES.values())[ti].split()) | {"really", "today", "nicely"}
    res = tp.fit_nmf(docs, toks, 4)
    reps = tp.representative_docs(res, docs, tp.evaluation_space(docs), 2)
    assert all(len(v) == 2 for v in reps.values())


def test_outliers_excluded_from_cluster_metrics(prepared):
    _, truth, docs, toks, keep = prepared
    space = tp.evaluation_space(docs)
    res = tp.fit_nmf(docs, toks, 4)
    res.labels = res.labels.copy()
    res.labels[:20] = -1
    s = tp.score_result(res, toks, space)
    assert s["noise_fraction"] == pytest.approx(20 / len(docs))


# ------------------------------------------------------------------------------- keywords
def kw_texts():
    texts, _ = corpus(60)
    return texts


@pytest.mark.parametrize("method", ["frequency", "tfidf", "rake", "yake", "textrank"])
def test_keyword_methods_return_planted_vocabulary(method):
    out = kw.METHODS[method](kw_texts(), top_n=15)
    assert 0 < len(out) <= 15
    planted = {w for v in THEMES.values() for w in v.split()}
    valid = planted | {"really", "today", "nicely"}          # the corpus vocabulary
    words = [w for k, _ in out for w in k.split()]
    assert all(w in valid for w in words), (method, out[:5])  # never a stopword or invented token
    assert sum(1 for w in words if w in planted) / len(words) >= 0.7, (method, out[:5])
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_rake_prefers_multiword_phrases_over_stopwords():
    texts = ["machine learning models need data. machine learning is fun.",
             "deep machine learning helps; data is useful."] * 3
    top = [k for k, _ in kw.rake_keywords(texts, 5)]
    assert any(k == "machine learning" for k in top) and "is" not in top


def test_textrank_ranks_hub_word_first():
    texts = ["alpha hub beta", "gamma hub delta", "epsilon hub zeta", "eta hub theta"] * 3
    assert kw.textrank_keywords(texts, 3)[0][0] == "hub"


def test_embedding_mmr_diversifies_and_needs_encoder():
    texts = kw_texts()
    vocab = sorted({w for v in THEMES.values() for w in v.split()})
    theme_of = {w: i for i, v in enumerate(THEMES.values()) for w in v.split()}

    def enc(ts):
        out = np.zeros((len(ts), 5)) + 1e-3
        for i, t in enumerate(ts):
            for w in t.split():
                if w in theme_of:
                    out[i, theme_of[w]] += 1
        return out
    plain = kw.embedding_mmr_keywords(texts, 8, encoder=enc, diversity=0.0)
    diverse = kw.embedding_mmr_keywords(texts, 8, encoder=enc, diversity=0.8)
    themes = lambda r: len({theme_of[k.split()[0]] for k, _ in r if k.split()[0] in theme_of})  # noqa: E731
    assert themes(diverse) >= themes(plain) and vocab
    with pytest.raises(ImportError):
        kw.embedding_mmr_keywords(texts, 5)


def test_run_all_reports_agreement_stability_and_errors():
    r = kw.run_all(kw_texts(), top_n=10, stability_runs=2)
    assert "embedding_mmr" in r["errors"] and "ImportError" in r["errors"]["embedding_mmr"]
    assert set(r["keywords"]) == {"frequency", "tfidf", "rake", "yake", "textrank"}
    assert all(0 <= v <= 1 for v in r["agreement_jaccard"].values())
    assert all(v is not None and 0 <= v <= 1 for v in r["stability_jaccard"].values())
    assert kw.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


def test_devanagari_and_urdu_words_are_never_split_by_vectorisers():
    """Regression: sklearn's default token pattern splits words at Devanagari vowel signs."""
    from chatlens.nlp.classifiers import tfidf_methods
    from chatlens.nlp.similarity import SimilarityIndex
    hi = ["बहुत अच्छा खाना", "बहुत स्वादिष्ट खाना", "आज बहुत अच्छा दिन"] * 4
    ur = ["بہت اچھا کھانا", "بہت مزیدار کھانا", "آج بہت اچھا دن"] * 4
    for corpus_ in (hi, ur):
        words = {w for k, _ in kw.tfidf_keywords(corpus_, 20) for w in k.split()}
        assert all(len(w) >= 3 for w in words) and words
        docs, toks, _ = tp.prepare_docs([" ".join(corpus_[:3])] * 3, min_tokens=2)
        assert all(len(t) >= 3 for d in toks for t in d)
    vec = tfidf_methods()[0]._vf().fit(hi)
    feats = set(vec.get_feature_names_out())
    assert {"बहुत", "अच्छा", "स्वादिष्ट"} <= feats                # whole words survive
    assert not feats & {"बह", "अच", "स्व", "ष्ट"}                   # fragments produced by the old default pattern
    assert SimilarityIndex().fit(hi).query("बहुत अच्छा खाना", 1)[0][1] > 0.9


def test_best_k_breaks_coherence_near_ties_with_silhouette():
    rows = [{"k": 3, "npmi_coherence": 0.305, "topic_diversity": 0.9, "silhouette": 0.23},     # drops a theme
            {"k": 4, "npmi_coherence": 0.299, "topic_diversity": 0.88, "silhouette": 0.30},
            {"k": 6, "npmi_coherence": 0.135, "topic_diversity": 0.58, "silhouette": 0.21}]
    assert tp.best_k(rows)["k"] == 4
    assert tp.best_k(rows, tolerance=0.0)["k"] == 3                    # strict coherence picks the lossy one
    assert tp.best_k([{"k": 2, "npmi_coherence": 0.9, "topic_diversity": 0.2, "silhouette": 0.9},
                      {"k": 5, "npmi_coherence": 0.3, "topic_diversity": 0.9, "silhouette": 0.1}])["k"] == 5
    assert tp.best_k([{"k": 2, "npmi_coherence": 0.1, "topic_diversity": 0.9, "silhouette": None}])["k"] == 2
