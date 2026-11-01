"""Heuristic per-message language/script identification.

Method (documented limitation): script-based, plus a Roman-Hindi/Urdu lexicon to separate
English from Hinglish in Latin script.
  * Arabic script  -> "ur"  (could also be Arabic/Persian/Pashto; not distinguished)
  * Devanagari     -> "hi"  (could also be Marathi/Nepali; not distinguished)
  * Latin script   -> "hinglish" if >= 2 lexicon hits and >= 25% of tokens, else "en"
                      (any other Latin-script language is labelled "en")
  * several scripts each >= 25% of letters -> "mixed"
  * fewer than 2 letters/tokens -> "und" (undetermined)
This is a transparent baseline, not a trained language-ID model; evaluate against labelled data
(see the NLP stage) before relying on it for research conclusions.
"""
from __future__ import annotations

from .stopwords import HINGLISH

_HINGLISH_LEX = HINGLISH - {"the", "to", "do", "ho", "me", "hi", "na", "he", "main", "tu"}


def _script(ch: str) -> str | None:
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF \
            or 0xFE70 <= o <= 0xFEFF:
        return "arabic"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if ch.isalpha() and o < 0x250:
        return "latin"
    return None


def detect_language(text: str, tokens: list[str] | None = None) -> str:
    counts = {"arabic": 0, "devanagari": 0, "latin": 0}
    for ch in text:
        s = _script(ch)
        if s:
            counts[s] += 1
    total = sum(counts.values())
    if total < 2:
        return "und"
    major = [k for k, v in counts.items() if v / total >= 0.25]
    if len(major) > 1:
        return "mixed"
    dom = max(counts, key=counts.get)
    if dom == "arabic":
        return "ur"
    if dom == "devanagari":
        return "hi"
    toks = tokens if tokens is not None else text.casefold().split()
    if len(toks) < 2:
        return "und"
    hits = sum(1 for t in toks if t in _HINGLISH_LEX)
    return "hinglish" if hits >= 2 and hits / len(toks) >= 0.25 else "en"
