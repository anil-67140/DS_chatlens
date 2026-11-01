import pytest

from chatlens import AnalysisConfig, IngestionError, parse_chat, preprocess, read_chat
from chatlens.analytics import behavior, overview, participants, temporal, text


def test_config_yaml_roundtrip_and_validation(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("session_gap_minutes: 30\nextra_media_markers: ['x']\n")
    c = AnalysisConfig.from_yaml(p)
    assert c.session_gap_minutes == 30 and c.extra_media_markers == ("x",)
    p.write_text("bogus_key: 1\n")
    with pytest.raises(ValueError, match="Unknown"):
        AnalysisConfig.from_yaml(p)
    with pytest.raises(ValueError):
        AnalysisConfig(session_gap_minutes=0)
    with pytest.raises(ValueError):
        AnalysisConfig(alpha=1.5)


def test_default_yaml_matches_dataclass_defaults():
    assert AnalysisConfig.from_yaml("configs/default.yaml") == AnalysisConfig()


def test_read_chat_from_path(tmp_path):
    f = tmp_path / "chat.txt"
    f.write_text("[01/01/2024, 10:00:00] A: hi", encoding="utf-8")
    assert "hi" in read_chat(f).text
    with pytest.raises(IngestionError, match="not found"):
        read_chat(tmp_path / "nope.txt")
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * 2_000_000)
    with pytest.raises(IngestionError, match="limit"):
        read_chat(big, config=AnalysisConfig(max_upload_mb=1))


def test_utf16_with_bom():
    raw = "[01/01/2024, 10:00:00] A: hi".encode("utf-16")
    assert "hi" in read_chat(raw, filename="a.txt").text


def test_corrupt_zip():
    with pytest.raises(IngestionError, match="Corrupt"):
        read_chat(b"PK\x03\x04garbage", filename="a.zip")


def test_chat_with_only_system_messages_does_not_crash():
    d = preprocess(parse_chat("[01/01/2024, 10:00:00] Messages and calls are end-to-end "
                              "encrypted.\n").df)
    o = overview.overview(d)
    assert o["total_messages"] == 0 and o["total_participants"] == 0
    assert temporal.streaks(d)["longest_days"] == 0
    assert temporal.hourly(d).sum() == 0
    assert participants.concentration(d) == {}
    assert behavior.response_summary(d)["counted_responses"] == 0
    assert behavior.exchange_stats(d) == {}
    assert behavior.no_observed_reply(d).empty
    assert text.top_words(d).empty and text.length_distribution(d) == {}
    assert temporal.seasonality(d)["reliable"] is False


def test_single_message_chat():
    d = preprocess(parse_chat("[01/01/2024, 10:00:00] A: hello there\n").df)
    assert overview.overview(d)["total_messages"] == 1
    assert behavior.response_times(d).empty
    assert participants.participant_table(d).loc["A", "messages"] == 1


def test_single_sender_chat_has_no_responses():
    txt = "\n".join(f"[01/01/2024, 10:0{i}:00] A: msg {i}" for i in range(5))
    d = preprocess(parse_chat(txt).df)
    assert behavior.response_summary(d)["turn_changes"] == 0
    assert len(behavior.turns(d)) == 1


def test_seasonality_reliable_only_with_two_years():
    from datetime import datetime, timedelta
    t0 = datetime(2022, 1, 1)
    txt = "\n".join(f"[{t0 + timedelta(days=30 * i):%d/%m/%Y}, 10:00:00] A: hi" for i in range(30))
    d = preprocess(parse_chat(txt).df)
    s = temporal.seasonality(d)
    assert s["reliable"] is True and s["months_span"] >= 24
    assert temporal.trend(d)["reliable"] in (True, False)


def test_invalid_timestamp_rows_excluded_from_time_analytics_only():
    d = preprocess(parse_chat("[01/01/2024, 10:00:00] A: ok\n[31/02/2024, 10:00:00] A: bad\n"
                              "[02/01/2024, 10:00:00] B: ok\n").df)
    assert overview.overview(d)["total_messages"] == 3      # still counted
    assert temporal.hourly(d).sum() == 2                    # not placed on a clock
    assert text.top_words(d, 5)["count"].sum() >= 1


def test_length_distribution_and_frequent_messages(chat_df):
    ld = text.length_distribution(chat_df)
    assert ld["histogram"]["messages"].sum() == chat_df["is_text"].sum()
    fm = text.frequent_messages(chat_df, min_count=20)
    assert fm.iloc[0]["count"] == 20
    assert not text.emoji_by_sender(chat_df).empty
    assert behavior.exchange_stats(chat_df)["sessions"] == 40
    assert behavior.response_distribution(chat_df)["responses"].sum() == 60
    assert not temporal.weekly(chat_df).empty and not temporal.monthly(chat_df).empty
    assert not participants.monthly_activity(chat_df).empty
