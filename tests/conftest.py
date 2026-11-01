"""Synthetic fixtures. All data here is SYNTHETIC and deterministic: values are chosen so the
correct answer of each analytic is known by construction."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from chatlens import parse_chat, preprocess


def build_events(days: int = 20, start: datetime = datetime(2024, 3, 1)):
    """Per day: A 10:00, B 10:05, A 10:08, B 10:13, A 23:00 (never answered)."""
    ev = []
    for d in range(days):
        day = start + timedelta(days=d)
        ev += [
            (day.replace(hour=10, minute=0), "Ali", "good morning bhai kya haal hai"),
            (day.replace(hour=10, minute=5), "Sara", "hello! all good 😀 https://www.example.com/a"),
            (day.replace(hour=10, minute=8), "Ali", "<Media omitted>"),
            (day.replace(hour=10, minute=13), "Sara", "nice one 😀😂"),
            (day.replace(hour=23, minute=0), "Ali", "night night"),
        ]
    return ev


def render_android_24h_dmy(ev) -> str:
    lines = ["31/12/23, 09:00 - Messages and calls are end-to-end encrypted. No one outside "
             "of this chat can read them."]
    lines += [f"{t:%d/%m/%y, %H:%M} - {s}: {m}" for t, s, m in ev]
    return "\n".join(lines) + "\n"


def render_android_12h_mdy(ev) -> str:
    out = []
    for t, s, m in ev:
        hh = t.strftime("%I").lstrip("0")
        out.append(f"{t.month}/{t.day}/{t:%y}, {hh}:{t:%M} {t:%p} - {s}: {m}")
    return "\n".join(out) + "\n"


def render_ios(ev) -> str:
    return "\n".join(f"\u200e[{t:%d/%m/%Y}, {t:%H:%M:%S}] {s}: {m}" for t, s, m in ev) + "\n"


@pytest.fixture(scope="session")
def events():
    return build_events()


@pytest.fixture(scope="session")
def chat_df(events):
    res = parse_chat(render_ios(events))
    return preprocess(res.df)


@pytest.fixture(scope="session")
def chat_result(events):
    return parse_chat(render_ios(events))
