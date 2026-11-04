import pandas as pd
import pytest

from chatlens import parse_chat, preprocess, validate
from chatlens.privacy import anonymize
from chatlens.validation import infer_chat_type


def test_clean_chat_passes(chat_df, chat_result):
    q = validate(chat_df, chat_result.report)
    assert q.ok and not [c for c in q.failed() if c.severity == "error"]


def test_future_and_ancient_timestamps_fail():
    d = parse_chat("[01/01/1999, 10:00:00] A: old\n[01/01/2099, 10:00:00] A: future\n").df
    q = validate(d)
    c = next(c for c in q.checks if c.name == "timestamps_plausible")
    assert not c.passed and c.count == 2 and not q.ok


def test_regression_warning():
    d = parse_chat("[02/01/2024, 10:00:00] A: x\n[01/01/2024, 10:00:00] A: y\n").df
    c = next(c for c in validate(d).checks if c.name == "chronological_order")
    assert not c.passed and c.severity == "warning"


def test_chat_type_inference(chat_df):
    assert infer_chat_type(chat_df) == "individual"
    g = parse_chat("[01/01/2024, 10:00:00] A: x\n[01/01/2024, 10:01:00] B: y\n"
                   "[01/01/2024, 10:02:00] C: z\n").df
    assert infer_chat_type(g) == "group"


def test_anonymize_senders_and_text():
    raw = parse_chat(
        "[01/01/2024, 10:00:00] Ali Khan: call me on +92 300 1234567 or ali@mail.com\n"
        "[01/01/2024, 10:01:00] Sara: ok Ali Khan see https://secret.example/path?token=abc123 now\n").df
    a, mapping = anonymize(raw)
    d = preprocess(a)
    assert set(d["sender"]) == {"Participant 1", "Participant 2"}
    joined = " ".join(d["message"]) + " ".join(map(str, d["tokens"])) + " ".join(d["text_clean"])
    for leak in ("Ali Khan", "ali", "1234567", "ali@mail.com", "abc123", "token", "/path?"):
        assert leak not in joined.lower() or leak == "ali" and "participant" in joined.lower(), leak
    assert d.loc[1, "domains"] == ["secret.example"]             # domain kept for link analytics
    assert mapping["Ali Khan"] == "Participant 1"
    assert raw.loc[0, "sender"] == "Ali Khan"                    # original frame untouched
    assert "participant" in d.loc[1, "tokens"]                   # text features are derived, not blanked


def test_anonymize_refuses_preprocessed_frames():
    with pytest.raises(ValueError, match="before preprocess"):
        anonymize(preprocess(parse_chat("[01/01/2024, 10:00:00] A: hi there\n").df))


def test_anonymize_system_rows_keep_missing_sender():
    raw = parse_chat("[01/01/2024, 10:00:00] Messages and calls are end-to-end encrypted.\n"
                     "[01/01/2024, 10:01:00] A: hi\n").df
    a, _ = anonymize(raw)
    assert pd.isna(a.loc[0, "sender"]) and a.loc[1, "sender"] == "Participant 1"


def test_logging_never_contains_message_text(caplog, events):
    import logging

    from conftest import render_ios
    with caplog.at_level(logging.INFO, logger="chatlens"):
        preprocess(parse_chat(render_ios(events)).df)
    assert "good morning" not in caplog.text and "stage=preprocess" in caplog.text
