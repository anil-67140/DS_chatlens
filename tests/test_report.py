import json
import re

import pandas as pd
import pytest

from chatlens.analyze import NLPOptions, analyze_chat
from chatlens.nlp.chat_nlp import LanguageRouter, vader_model
from chatlens.report import build_report_html, export_bundle
from chatlens.synthetic import make_synthetic_chat

XSS = '<script>alert(1)</script>'


@pytest.fixture(scope="module")
def bundle():
    text, _ = make_synthetic_chat(1500, seed=2)
    text = text.replace("Ali:", f"{XSS}:").replace("Sara:", "<img src=x onerror=alert(2)>:")
    return analyze_chat(text, nlp=NLPOptions(sentiment=LanguageRouter({"en": vader_model()}), topic_ks=(3,)))


def test_report_escapes_hostile_names_everywhere(bundle):
    html = build_report_html(bundle)
    assert "<script>alert(1)</script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html or "\\u003cscript\\u003ealert(1)" in html
    # the only <script> tags are Plotly's own
    for tag in re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S):
        assert "alert(1)" not in tag.replace("\\u003c", "<") or "\\u003c" in tag or "&lt;" in tag


def test_report_contains_required_sections(bundle):
    html = build_report_html(bundle, benchmark={"toy": pd.DataFrame({"method": ["a"], "macro_f1": [0.5]})})
    for needle in ("Parsing and data quality", "Participants", "Conversation behaviour", "Statistical tests",
                   "Methodology and limitations", "Model benchmark", "not 'ignored'", "Significance ≠ importance"):
        assert needle in html, needle


def test_export_omits_message_text_by_default(bundle, tmp_path):
    files = export_bundle(bundle, tmp_path)
    csv = (tmp_path / "processed_messages.csv").read_text()
    assert "good morning" not in csv and "message" not in csv.splitlines()[0].split(",")
    assert json.loads((tmp_path / "summary.json").read_text())["meta"]["n_rows"] == len(bundle.df)
    assert {"report", "processed_messages", "summary", "participants"} <= set(files)
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_export_with_text_is_explicit_opt_in(bundle, tmp_path):
    export_bundle(bundle, tmp_path, include_text=True)
    assert "message" in (tmp_path / "processed_messages.csv").read_text().splitlines()[0]
