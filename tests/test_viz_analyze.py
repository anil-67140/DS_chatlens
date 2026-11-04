import pytest

from chatlens.analyze import NLPOptions, analyze_chat, message_level_tests, session_documents
from chatlens.config import AnalysisConfig
from chatlens.nlp.chat_nlp import LabelModel, LanguageRouter, vader_model
from chatlens.synthetic import make_synthetic_chat
from chatlens.viz import ChartSpec, chart_specs, make_wordcloud, render_mpl, render_plotly, save_all


@pytest.fixture(scope="module")
def bundle():
    text, _ = make_synthetic_chat(2500, seed=11)
    return analyze_chat(text, nlp=NLPOptions(sentiment=LanguageRouter({"en": vader_model()}), topic_ks=(3, 5)))


def test_bundle_has_all_sections_and_consistent_counts(bundle):
    b = bundle
    assert b.meta["chat_type"] == "group" and b.quality["ok"] and b.parse_report["unparsed_leading_lines"] == 0
    assert b.overview["total_messages"] == b.participants["messages"].sum()
    assert set(b.timings) >= {"parse", "preprocess", "analytics", "sentiment", "topics", "keywords"}
    assert b.nlp["topics"]["n_sessions_used"] > 60 and len(b.nlp["topics"]["topics"]) == b.nlp["topics"]["k"]
    assert "tfidf" in b.nlp["keywords"]["keywords"] and b.nlp["summaries"]
    assert "summary" not in b.nlp["summaries"][0]                  # message text not exposed by default


def test_sentiment_routing_reports_unscored_instead_of_guessing(bundle):
    s = bundle.nlp["sentiment"]
    assert s["coverage_by_language"]["en"] == 1.0
    for lang in ("ur", "hi", "hinglish"):
        assert s["coverage_by_language"].get(lang, 0.0) == 0.0      # English-only model never scores them
    assert 0.3 < s["coverage"] < 0.8 and "caveat" in s and "monitoring" in s
    sc = bundle.nlp["sentiment_scores"]
    assert (sc["label"] == "unscored").sum() == s["n_text_messages"] - s["n_scored"]


def test_anonymised_bundle_has_no_names_anywhere():
    text, truth = make_synthetic_chat(1500, seed=5)
    b = analyze_chat(text, anonymize_names=True, nlp=NLPOptions(run_topics=False))
    assert set(b.participants.index) == {f"Participant {i}" for i in range(1, 6)}
    blob = " ".join(map(str, b.df["message"])) + " ".join(map(str, b.participants.index))
    assert not any(n in blob for n in truth.participants)
    assert "name_mapping" in b.private and b.meta["anonymized"]


def test_topics_skipped_with_reason_on_tiny_chat():
    b = analyze_chat("[01/01/2024, 10:00:00] A: hi there friend\n[01/01/2024, 10:01:00] B: hello again\n",
                     nlp=NLPOptions(run_summaries=False))
    assert "skipped" in b.nlp["topics"] and "skipped" in b.nlp["keywords"]


def test_session_documents_one_per_session(bundle):
    docs, ids = session_documents(bundle.df, AnalysisConfig())
    assert len(docs) == len(ids) and len(set(ids)) == len(ids)


def test_message_level_tests_collects_effect_sizes(bundle):
    t = message_level_tests(bundle)
    k = t["participant_comparison"]["message_length_by_sender"]
    assert k["magnitude"] in {"negligible", "small", "medium", "large"} and "epsilon" in k["effect_name"]


def test_chart_specs_render_in_both_engines(bundle):
    specs = chart_specs(bundle)
    assert {"activity_heatmap", "hourly", "timeline", "participants", "response_times", "top_words", "emoji",
            "sentiment_participants", "languages"} <= set(specs)
    import matplotlib.pyplot as plt
    for name, sp in specs.items():
        assert render_plotly(sp).data, name
        fig = render_mpl(sp)
        assert fig.axes, name
        plt.close(fig)
    with pytest.raises(ValueError):
        render_plotly(ChartSpec("x", "pie", "t"))


def test_save_all_writes_files_atomically(bundle, tmp_path):
    w = save_all(bundle, tmp_path)
    assert len(w["html"]) == len(w["png"]) > 8
    assert all((tmp_path / "png" / f"{n}.png").read_bytes()[:4] == b"\x89PNG" for n in chart_specs(bundle))
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_wordcloud_handles_empty_and_latin():
    assert make_wordcloud({}) is None
    assert make_wordcloud({"hello": 5, "world": 3}) is not None


def test_label_model_without_probabilities():
    class M:
        classes_ = ["a", "b"]

        def predict(self, texts):
            return ["a"] * len(texts)
    labels, conf = LabelModel("m", M()).predict_label(["x", "y"])
    assert list(labels) == ["a", "a"] and conf is None


@pytest.mark.parametrize("text", [
    "[01/01/2024, 10:00:00] Messages and calls are end-to-end encrypted.\n",                 # only system notices
    "[01/01/2024, 10:00:00] A: hi\n",                                                          # one message
    "\n".join(f"[01/01/2024, 10:0{i}:00] A: msg {i}" for i in range(6)),                       # one sender
    "[31/02/2024, 10:00:00] A: impossible\n[31/02/2024, 10:01:00] B: dates\n",                 # all timestamps invalid
    "[01/01/2024, 10:00:00] A: <Media omitted>\n[01/01/2024, 10:01:00] B: <Media omitted>\n",  # media only
])
def test_full_pipeline_never_crashes_on_degenerate_chats(text):
    """Regression: an empty object-dtype mask used to raise KeyError deep inside behaviour analytics."""
    from chatlens.report import build_report_html
    b = analyze_chat(text)
    assert isinstance(b.overview["total_messages"], int)
    assert "<html" in build_report_html(b)
    chart_specs(b)


def test_anonymised_degenerate_chat_does_not_crash():
    b = analyze_chat("[01/01/2024, 10:00:00] Messages and calls are end-to-end encrypted.\n", anonymize_names=True)
    assert b.overview["total_messages"] == 0
