"""Preprocessing pipeline: enriches a parsed DataFrame with analysis-ready columns."""
from __future__ import annotations

import pandas as pd

from ..config import AnalysisConfig
from ..logging_utils import get_logger, timed
from . import stopwords as sw
from .extract import domain_of, extract_emojis, extract_urls
from .language import detect_language
from .normalize import normalize_text, tokenize

log = get_logger()

ADDED_COLUMNS = ["is_text", "urls", "domains", "emojis", "text_clean", "tokens", "tokens_content",
                 "language", "n_words", "n_chars", "n_emojis"]

_EMPTY: list = []


def _features(msg: str, cache: dict, include_noise: bool, extra: frozenset[str]) -> tuple:
    """All per-message features in one pass (avoids repeated column scans)."""
    urls = extract_urls(msg)
    domains = [d for d in (domain_of(u) for u in urls) if d]
    emojis = extract_emojis(msg)
    clean = normalize_text(msg)
    toks = tokenize(clean)
    lang = detect_language(clean, toks)
    if lang not in cache:
        cache[lang] = sw.for_language(lang, include_chat_noise=include_noise, extra=extra)
    stop = cache[lang]
    content = [t for t in toks if t not in stop and len(t) > 1]
    return urls, domains, emojis, clean, toks, content, lang


def preprocess(df: pd.DataFrame, config: AnalysisConfig | None = None,
               extra_stopwords: frozenset[str] = frozenset(),
               include_chat_noise: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` with text features. No rows are removed."""
    _ = config or AnalysisConfig()
    out = df.copy()
    with timed(log, "preprocess") as info:
        is_text = ~(out["is_system"] | out["is_media"] | out["is_deleted"] | out["is_empty"])
        out["is_text"] = is_text
        cols: dict[str, list] = {k: [] for k in (
            "urls", "domains", "emojis", "text_clean", "tokens", "tokens_content", "language")}
        n_chars: list[int] = []
        cache: dict[str, frozenset[str]] = {}
        for msg, ok in zip(out["message"].fillna("").tolist(), is_text.tolist(), strict=True):
            if ok:
                urls, domains, emojis, clean, toks, content, lang = _features(
                    msg, cache, include_chat_noise, extra_stopwords)
                n_chars.append(len(msg))
            else:
                urls = domains = emojis = toks = content = _EMPTY
                clean, lang = "", "und"
                n_chars.append(0)
            for k, v in zip(cols, (urls, domains, emojis, clean, toks, content, lang),
                            strict=True):
                cols[k].append(v)
        for k, v in cols.items():
            out[k] = v
        out["n_emojis"] = [len(e) for e in cols["emojis"]]
        # Word counts exclude URLs/emoji/mentions: they count what a person typed as words.
        out["n_words"] = [len(t) for t in cols["tokens"]]
        out["n_chars"] = n_chars
        info["rows"] = len(out)
    return out
