from datetime import datetime

import pandas as pd
import pytest
from conftest import render_android_12h_mdy, render_android_24h_dmy, render_ios

from chatlens import ParseError, parse_chat


def test_formats_produce_identical_messages(events):
    frames = {}
    for name, fn in [("ios", render_ios), ("a24", render_android_24h_dmy),
                     ("a12", render_android_12h_mdy)]:
        res = parse_chat(fn(events))
        df = res.df[~res.df["is_system"]]
        frames[name] = df
        assert len(df) == len(events), name
    for name, df in frames.items():
        assert df["timestamp"].tolist() == [e[0] for e in events], name
        assert df["sender"].tolist() == [e[1] for e in events], name


def test_date_order_detection(events):
    assert parse_chat(render_android_24h_dmy(events)).report.date_order == "day-first"
    assert parse_chat(render_android_12h_mdy(events)).report.date_order == "month-first"


def test_ambiguous_dates_flagged():
    text = "01/02/24, 10:00 - A: hi\n01/02/24, 10:05 - B: hey\n"
    rep = parse_chat(text).report
    assert rep.date_order_confident is False
    assert any("ambiguous" in w for w in rep.warnings)


def test_monotonic_heuristic_resolves_ambiguity():
    # 03/04 -> 04/04 -> 05/04: only month-first? No: day-first gives 3,4,5 April (monotonic).
    text = "03/04/24, 10:00 - A: a\n04/04/24, 10:00 - B: b\n05/04/24, 10:00 - A: c\n"
    res = parse_chat(text)
    assert res.report.date_order == "day-first"


def test_multiline_message_merged():
    text = "[01/01/2024, 10:00:00] A: line1\nline2\n\nline4\n[01/01/2024, 10:01:00] B: ok\n"
    res = parse_chat(text)
    assert len(res.df) == 2
    assert res.df.loc[0, "message"] == "line1\nline2\n\nline4"
    assert res.report.multiline_messages == 1


def test_system_messages_detected(events):
    res = parse_chat(render_android_24h_dmy(events))
    assert res.report.system_messages == 1
    assert res.df.loc[0, "is_system"] and pd.isna(res.df.loc[0, "sender"])


def test_group_events_are_system_even_with_colon_in_group_name():
    text = '1/1/24, 10:00 - Ali created group "Team: Alpha"\n1/1/24, 10:01 - Ali: added you\n'
    res = parse_chat(text)
    assert res.df.loc[0, "is_system"]
    assert not res.df.loc[1, "is_system"] and res.df.loc[1, "sender"] == "Ali"


@pytest.mark.parametrize("marker", ["<Media omitted>", "image omitted", "sticker omitted",
                                    "<attached: 00001-PHOTO-2024.jpg>",
                                    "IMG-20240101-WA0001.jpg (file attached)"])
def test_media_markers(marker):
    res = parse_chat(f"[01/01/2024, 10:00:00] A: {marker}\n")
    assert res.df.loc[0, "is_media"], marker


def test_extra_media_marker_via_config():
    from chatlens import AnalysisConfig
    cfg = AnalysisConfig(extra_media_markers=("میڈیا شامل نہیں",))
    res = parse_chat("[01/01/2024, 10:00:00] A: میڈیا شامل نہیں\n", cfg)
    assert res.df.loc[0, "is_media"]


def test_deleted_and_edited_flags():
    res = parse_chat("[01/01/2024, 10:00:00] A: This message was deleted\n"
                     "[01/01/2024, 10:01:00] A: fixed typo <This message was edited>\n")
    assert res.df.loc[0, "is_deleted"] and res.df.loc[1, "is_edited"]
    assert res.df.loc[1, "message"] == "fixed typo"


def test_invalid_timestamp_kept_not_dropped():
    text = ("[01/01/2024, 10:00:00] A: ok\n[31/02/2024, 10:00:00] A: impossible date\n"
            "[02/01/2024, 10:00:00] B: ok\n")
    res = parse_chat(text)
    assert len(res.df) == 3
    assert res.report.invalid_timestamps == 1 and res.df["timestamp"].isna().sum() == 1


def test_leading_garbage_reported():
    res = parse_chat("junk line\n[01/01/2024, 10:00:00] A: hi\n")
    assert res.report.unparsed_leading_lines == 1 and len(res.df) == 1


def test_duplicates_flagged_not_removed():
    res = parse_chat("[01/01/2024, 10:00:00] A: same\n[01/01/2024, 10:00:00] A: same\n")
    assert len(res.df) == 2 and res.df["is_duplicate"].tolist() == [False, True]


def test_not_a_chat_raises():
    with pytest.raises(ParseError):
        parse_chat("this is just some random text\nwithout headers\n")


def test_unicode_and_invisible_chars():
    res = parse_chat("\u200e[01/01/2024, 9:41:00\u202fPM] \u202aAli\u202c: \u200eسلام 🙂\n")
    assert res.df.loc[0, "timestamp"] == datetime(2024, 1, 1, 21, 41)
    assert res.df.loc[0, "sender"] == "Ali" and "سلام" in res.df.loc[0, "message"]


def test_midnight_and_noon_12h():
    res = parse_chat("1/5/24, 12:00 AM - A: x\n1/5/24, 12:30 PM - A: y\n1/5/24, 11:59 PM - A: z\n")
    assert [t.hour for t in res.df["timestamp"]] == [0, 12, 23]


def test_row_accounting_invariant(events):
    res = parse_chat(render_android_24h_dmy(events))
    r = res.report
    assert r.messages == r.header_lines  # every header became exactly one row
    assert r.total_lines >= r.header_lines
