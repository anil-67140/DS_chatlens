"""Synthetic WhatsApp chat generator.

PURPOSE: demo, tests and load-testing without anyone's private data. Everything produced here is
FABRICATED: the temporal and behavioural patterns are generated from the parameters below, so
analysing them reveals the generator, not human behaviour. ``truth`` records the generating
parameters so tests can check the pipeline recovers them. Message *text* can come from a real
public corpus (e.g. the Kaggle Roman Urdu / emotion sets) via ``phrase_bank``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

DEFAULT_PHRASES = {
    "en": ["good morning everyone", "are we meeting tomorrow", "that sounds great to me", "I will be late today",
           "please send the notes", "thanks a lot for the help", "what time is the meeting", "this is so funny",
           "happy birthday to you", "can you share the link", "I am not sure about this plan",
           "the weather is terrible today", "see you at the cafe", "did you finish the assignment"],
    "roman": ["kya haal hai yaar", "aaj kya plan hai", "bohot acha laga sun kar", "mujhe nahi pata kal kya hoga",
              "theek hai phir milte hain", "wah kya baat hai", "kab aa rahe ho", "ye to bohot bura hua",
              "shukriya bhai", "main abhi nikal raha hoon", "chai pe milte hain shaam ko"],
    "urdu": ["آپ کیسے ہیں", "کل ملتے ہیں", "بہت شکریہ", "یہ بہت اچھا ہے"],
    "hindi": ["आप कैसे हैं", "कल मिलते हैं", "बहुत धन्यवाद", "यह बहुत अच्छा है"],
}
EMOJIS = ["😀", "😂", "👍", "❤️", "🙏", "😍", "🔥", "😢", "🎉"]
DOMAINS = ["youtube.com", "github.com", "news.example.org", "docs.google.com"]
SYSTEM_START = "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, " \
               "can read or listen to them."


@dataclass
class SyntheticTruth:
    participants: list[str]
    target_share: dict[str, float]
    peak_hour: int
    n_sessions: int
    median_gap_minutes: float
    seed: int
    fabricated: bool = True
    notes: str = "FABRICATED data: patterns are generator parameters, not human behaviour."
    realised: dict = field(default_factory=dict)


def _diurnal(peak: int) -> np.ndarray:
    h = np.arange(24)
    d = np.exp(-0.5 * (((h - peak + 12) % 24 - 12) / 3.0) ** 2) + 0.03
    return d / d.sum()


def render(events: list[tuple[datetime, str | None, str]], fmt: str = "android_24h") -> str:
    out = []
    for t, who, msg in events:
        if fmt == "ios":
            head = f"\u200e[{t:%d/%m/%Y}, {t:%H:%M:%S}] "
        elif fmt == "android_12h":
            head = f"{t.month}/{t.day}/{t:%y}, {t.strftime('%I').lstrip('0')}:{t:%M}\u202f{t:%p} - "
        else:
            head = f"{t:%d/%m/%y}, {t:%H:%M} - "
        out.append(head + (f"{who}: {msg}" if who else msg))
    return "\n".join(out) + "\n"


def make_synthetic_chat(n_messages: int = 3000, participants: tuple[str, ...] = ("Ali", "Sara", "Bilal", "Ayesha",
                                                                                   "Usman"),
                        shares: tuple[float, ...] | None = None, days: int = 120,
                        start: datetime = datetime(2024, 1, 1), seed: int = 42, peak_hour: int = 21,
                        median_gap_minutes: float = 2.0, fmt: str = "android_24h",
                        phrase_bank: dict[str, list[str]] | None = None,
                        language_mix: dict[str, float] | None = None) -> tuple[str, SyntheticTruth]:
    rng = np.random.default_rng(seed)
    k = len(participants)
    p = np.array(shares if shares else [0.35, 0.25, 0.2, 0.12, 0.08][:k] if k <= 5 else np.ones(k), float)
    p = p / p.sum()
    bank = phrase_bank or DEFAULT_PHRASES
    mix = language_mix or {"en": 0.5, "roman": 0.4, "urdu": 0.05, "hindi": 0.05}
    langs = [lg for lg in mix if lg in bank]
    lp = np.array([mix[lg] for lg in langs])
    lp = lp / lp.sum()
    diur = _diurnal(peak_hour)

    events: list[tuple[datetime, str | None, str]] = [(start, None, SYSTEM_START)]
    if k > 2:
        events.append((start + timedelta(minutes=1), None, f"{participants[0]} created group \"Synthetic Demo\""))
    sessions = 0
    sigma = 0.8
    while sum(1 for e in events if e[1]) < n_messages:
        day = int(rng.integers(0, days))
        hour = int(rng.choice(24, p=diur))
        t = start + timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))
        if t.weekday() >= 5 and rng.random() < 0.3:      # slightly fewer weekend sessions
            continue
        sessions += 1
        who = int(rng.choice(k, p=p))
        for _ in range(int(rng.geometric(0.15))):
            lg = langs[int(rng.choice(len(langs), p=lp))]
            msg = str(rng.choice(bank[lg]))
            r = rng.random()
            if r < 0.12:
                msg += " " + "".join(rng.choice(EMOJIS, int(rng.integers(1, 3))))
            elif r < 0.17:
                msg += f" https://www.{rng.choice(DOMAINS)}/p/{int(rng.integers(1, 999))}"
            elif r < 0.22:
                msg = "<Media omitted>"
            elif r < 0.25:
                msg += "\nsecond line of the same message"
            events.append((t, participants[who], msg))
            t = t + timedelta(minutes=max(1, int(rng.lognormal(np.log(median_gap_minutes), sigma))))
            if rng.random() < 0.8:                        # usually someone else replies
                others = [i for i in range(k) if i != who]
                po = p[others] / p[others].sum()
                who = int(rng.choice(others, p=po))
            if t - events[-1][0] > timedelta(minutes=40):
                break
    head, body = events[:2 if k > 2 else 1], sorted(events[2 if k > 2 else 1:], key=lambda e: e[0])
    body = body[:n_messages + 5]
    text = render(head + body, fmt)
    truth = SyntheticTruth(list(participants), dict(zip(participants, p.round(4).tolist(), strict=True)),
                           peak_hour, sessions, median_gap_minutes, seed)
    return text, truth
