"""Analytics are verified against ground truth built into the synthetic chat:
20 days x [A 10:00, B 10:05, A 10:08, B 10:13, A 23:00]. B replies in 5 min, A in 3 min,
A's 23:00 message is never answered."""
import numpy as np
import pandas as pd
import pytest

from chatlens import AnalysisConfig
from chatlens.analytics import behavior, overview, participants, temporal, text

DAYS = 20


def test_overview_counts(chat_df):
    o = overview.overview(chat_df)
    assert o["total_messages"] == DAYS * 5
    assert o["total_participants"] == 2
    assert o["media_count"] == DAYS
    assert o["link_count"] == DAYS
    assert o["span_days"] == DAYS
    assert o["growth_daily"]["cumulative"].iloc[-1] == DAYS * 5
    assert "text" not in o["longest_message"]           # privacy default
    assert "text" in overview.overview(chat_df, include_text=True)["longest_message"]


def test_participant_table(chat_df):
    t = participants.participant_table(chat_df)
    assert t.loc["Ali", "messages"] == DAYS * 3 and t.loc["Sara", "messages"] == DAYS * 2
    assert t["pct_of_messages"].sum() == pytest.approx(100)
    assert t.loc["Sara", "links"] == DAYS and t.loc["Sara", "emojis"] == DAYS * 3  # 1 + 2
    assert t.loc["Ali", "media"] == DAYS


def test_concentration_known_values(chat_df):
    c = participants.concentration(chat_df)
    # shares 0.6/0.4 -> HHI = .36+.16 = .52 ; Gini of [60,40] = 0.1
    assert c["hhi"] == pytest.approx(0.52) and c["gini"] == pytest.approx(0.1)
    assert c["top1_share"] == pytest.approx(0.6)


def test_temporal_basics(chat_df):
    h = temporal.hourly(chat_df)
    assert h[10] == DAYS * 4 and h[23] == DAYS and h.sum() == DAYS * 5
    assert temporal.top_hours(chat_df, 1) == [10]
    assert temporal.heatmap(chat_df).to_numpy().sum() == DAYS * 5
    assert temporal.weekday(chat_df).sum() == DAYS * 5


def test_streak_and_silence(chat_df):
    s = temporal.streaks(chat_df)
    assert s["longest_days"] == DAYS and s["current_days"] == DAYS
    gaps = temporal.longest_inactive_periods(chat_df, 1)
    assert gaps.loc[0, "gap"] == pd.Timedelta(hours=12, minutes=47)   # 10:13 -> 23:00


def test_streak_with_gap():
    from datetime import datetime, timedelta

    from chatlens import parse_chat, preprocess
    days = [0, 1, 2, 5, 6]
    txt = "\n".join(f"[{(datetime(2024, 1, 1) + timedelta(days=d)):%d/%m/%Y}, 10:00:00] A: hi"
                    for d in days)
    s = temporal.streaks(preprocess(parse_chat(txt).df))
    assert s["longest_days"] == 3 and s["current_days"] == 2


def test_weekend_vs_weekday_uses_zero_days(chat_df):
    r = temporal.weekend_vs_weekday(chat_df)
    # constant 5/day -> means equal, negligible effect
    assert r["weekend_mean_per_day"] == pytest.approx(5) == r["weekday_mean_per_day"]
    assert r["test"] is None or r["test"]["significant"] is False


def test_trend_flat_series_not_significant(chat_df):
    t = temporal.trend(chat_df)
    assert t["reliable"] is False  # only 20 days < 8 full weeks


def test_response_times_ground_truth(chat_df):
    cfg = AnalysisConfig(bootstrap_iterations=200)
    s = behavior.response_summary(chat_df, cfg)
    assert s["by_responder"]["Sara"]["median_min"] == 5
    assert s["by_responder"]["Ali"]["median_min"] == 3
    # per day: 3 within-session replies (B,A,B); next-day 'Ali after Ali' is not a response
    assert s["counted_responses"] == DAYS * 3
    # Ali's 23:00 message follows Sara's 10:13 reply by ~10.8 h: a turn change outside the window
    assert s["excluded_beyond_window"] == DAYS
    lo, hi = s["by_responder"]["Sara"]["median_ci95"]
    assert lo <= 5 <= hi


def test_no_observed_reply_excludes_final_turn(chat_df):
    r = behavior.no_observed_reply(chat_df).set_index("sender")
    # Ali: 23:00 turns on days 1..19 unanswered; day 20 is the final turn -> excluded
    assert r.loc["Ali", "no_observed_reply"] == DAYS - 1
    # Sara: her 10:13 turn is followed only by Ali's 23:00 message, ~10.8 h later (> 6 h window)
    assert r.loc["Sara", "no_observed_reply"] == DAYS


def test_sessions_and_initiation(chat_df):
    s = behavior.sessions_summary(chat_df)
    assert len(s) == DAYS * 2                      # 10:00 block and 23:00 message per day
    init = behavior.initiation(chat_df).set_index("sender")
    assert init.loc["Ali", "sessions_started"] == DAYS * 2 and "Sara" not in init.index


def test_response_window_is_configurable(chat_df):
    s = behavior.response_summary(chat_df, AnalysisConfig(response_window_minutes=4,
                                                          bootstrap_iterations=100))
    assert s["by_responder"].get("Ali", {}).get("median_min") == 3
    assert "Sara" not in s["by_responder"]         # 5-min replies fall outside a 4-min window


def test_interaction_matrix(chat_df):
    m = behavior.interaction_matrix(chat_df)
    assert m.loc["Sara", "Ali"] == DAYS * 2 and m.loc["Ali", "Sara"] == DAYS


def test_text_analytics(chat_df):
    w = text.top_words(chat_df, 5).set_index("word")["count"]
    assert w["morning"] == DAYS and "the" not in w.index
    e = text.emoji_frequency(chat_df).set_index("emoji")["count"]
    assert e["😀"] == DAYS * 2 and e["😂"] == DAYS
    assert text.domain_frequency(chat_df).iloc[0].tolist() == ["example.com", DAYS]
    bg = text.ngrams(chat_df, 2, min_count=DAYS)
    assert "good morning" in bg["ngram"].tolist()
    assert (text.language_distribution(chat_df)["pct"].sum()) == pytest.approx(100)


def test_ngrams_never_cross_messages():
    from chatlens import parse_chat, preprocess
    d = preprocess(parse_chat("[01/01/2024, 10:00:00] A: alpha\n[01/01/2024, 10:01:00] A: beta\n"
                              * 1).df)
    assert text.ngrams(d, 2, min_count=1).empty


def test_distinctive_vocabulary_finds_planted_word():
    from chatlens import parse_chat, preprocess
    rng = np.random.default_rng(0)
    lines = []
    for i in range(200):
        who = "A" if i % 2 == 0 else "B"
        word = "zebra" if who == "A" and rng.random() < 0.8 else "common"
        lines.append(f"[01/01/2024, {10 + i // 60:02d}:{i % 60:02d}:00] {who}: {word} talk")
    d = preprocess(parse_chat("\n".join(lines)).df)
    dv = text.distinctive_vocabulary(d, min_total=5, top=3)
    top_a = dv[dv["sender"] == "A"].iloc[0]
    assert top_a["word"] == "zebra" and top_a["log_odds_z"] > 1.96


def test_compare_participants_runs(chat_df):
    c = participants.compare_participants(chat_df)
    assert set(c) == {"message_length_by_sender", "time_of_day_by_sender"}
