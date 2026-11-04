import numpy as np
import pandas as pd
import pytest

from chatlens import parse_chat, preprocess
from chatlens.nlp import ner, summarize
from chatlens.nlp.similarity import SimilarityIndex
from chatlens.nlp.toxicity import ToxicityScorer


def test_extractive_summary_picks_representative_and_non_redundant():
    msgs = ["we should book the venue for saturday evening",
            "yes the venue for saturday evening works",
            "book the venue for saturday evening please",
            "lol", "ok",
            "budget for the venue is five thousand rupees",
            "random cat video here look at it now"]
    idx = summarize.extractive_summary(msgs, k=2)
    assert len(idx) == 2 and idx == sorted(idx) and 3 not in idx and 4 not in idx
    assert not {0, 1, 2} <= set(idx)                      # MMR avoids three near-identical lines


def test_summary_edge_cases():
    assert summarize.extractive_summary(["hi", "ok"], 3) == []
    assert summarize.extractive_summary(["one two three four"], 3) == [0]


def test_summarize_chat_privacy_default(chat_df):
    s = summarize.summarize_chat(chat_df, top_sessions=2, k=1)
    assert len(s) == 2 and "summary" not in s[0] and len(s[0]["summary_msg_ids"]) == 1
    assert "summary" in summarize.summarize_chat(chat_df, top_sessions=1, include_text=True)[0]


def test_similarity_tfidf_finds_spelling_variants():
    texts = ["wah kya baat hai", "wah kia bat hai", "mausam bohot kharab hai", "how are you today"]
    idx = SimilarityIndex().fit(texts)
    top = idx.query("wah kya baat hai", k=2, exclude_self=True)
    assert top[0][0] == 1 and top[0][1] > 0.3
    assert top[0][1] > idx.query("wah kya baat hai", k=4)[3][1]


def test_near_duplicates_and_dense_backend():
    texts = ["see you at 5pm", "See you at 5pm!", "completely different message"]
    pairs = SimilarityIndex().fit(texts).near_duplicates(0.9)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    enc = lambda ts: np.array([[len(t), t.count("a") + 1] for t in ts], float)  # noqa: E731
    d = SimilarityIndex("dense", enc).fit(texts)
    assert d.query("x", 1)
    with pytest.raises(ValueError):
        SimilarityIndex("dense")


class FakeDoc:
    def __init__(self, ents):
        self.ents = [type("E", (), {"text": t, "label_": lab}) for t, lab in ents]


class FakeNLP:
    def pipe(self, texts, batch_size=256):
        for t in texts:
            yield FakeDoc([(w, "PERSON") for w in t.split() if w.istitle()] + [("tomorrow", "DATE")])


def test_ner_counts_and_label_filter():
    df = ner.extract_entities(["Ali met Sara", "Ali left"], nlp=FakeNLP(), labels={"PERSON"})
    assert df.iloc[0].to_dict() == {"entity": "Ali", "label": "PERSON", "count": 2}
    assert "DATE" not in set(df["label"])


def test_ner_missing_model_message():
    with pytest.raises(ImportError, match="spacy"):
        ner.load_spacy("definitely_not_a_model")


def test_toxicity_requires_consent_and_scores_locally():
    with pytest.raises(PermissionError):
        ToxicityScorer()
    fake = lambda: (lambda texts: [[{"label": "toxic", "score": 0.9 if "idiot" in t else 0.05},  # noqa: E731
                                    {"label": "non-toxic", "score": 0.5}] for t in texts])
    s = ToxicityScorer(consent=True, pipeline_factory=fake)
    assert s.flag(["you idiot", "hello"]).tolist() == [True, False]
    assert s.score(["hello"])[0] == pytest.approx(0.05)


def test_ner_on_real_pipeline_objects_types():
    assert isinstance(preprocess(parse_chat("[01/01/2024, 10:00:00] A: hi\n").df), pd.DataFrame)
