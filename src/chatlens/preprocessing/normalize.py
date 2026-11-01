"""Unicode-aware text normalisation and tokenisation."""
from __future__ import annotations

import re
import unicodedata

import emoji

from ..parsing.patterns import URL_RE

# Arabic-script diacritics, tatweel; map Arabic yeh/kaf variants onto the Urdu forms.
_AR_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u0640]")
_AR_MAP = str.maketrans({"ك": "ک", "ي": "ی", "ى": "ی", "ې": "ی"})
_MENTION_RE = re.compile(r"@\u2068?\+?\d{6,15}\u2069?|@\d{6,15}")
_WORD_CHARS = r"[\w\u0900-\u0963\u0966-\u097f\u0610-\u061a\u064b-\u065f\u0670]"
TOKEN_RE = re.compile(rf"{_WORD_CHARS}+(?:['’]{_WORD_CHARS}+)?", re.UNICODE)


def normalize_text(text: str) -> str:
    """Lower-case, strip URLs/mentions/emoji/diacritics; keep letters and digits (all scripts)."""
    t = unicodedata.normalize("NFC", text)
    t = URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = emoji.replace_emoji(t, replace=" ")
    t = _AR_DIACRITICS.sub("", t).translate(_AR_MAP)
    return t.casefold()


def tokenize(text: str) -> list[str]:
    """Tokenise normalised text. Underscores and pure-digit tokens are dropped."""
    return [tok.strip("_") for tok in TOKEN_RE.findall(text)
            if tok.strip("_") and not tok.strip("_").isdigit()]
