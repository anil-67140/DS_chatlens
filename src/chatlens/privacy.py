"""Anonymisation and masking helpers. Deterministic (by first appearance), no external calls."""
from __future__ import annotations

import re

import pandas as pd

_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s\-().]{7,}\d(?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def participant_mapping(df: pd.DataFrame) -> dict[str, str]:
    order = df.loc[~df["is_system"], "sender"].dropna().drop_duplicates().tolist()
    return {name: f"Participant {i + 1}" for i, name in enumerate(order)}


_URL_RE = re.compile(r"(?:https?://|www\.)([^\s/?#]+)[^\s]*", re.IGNORECASE)
_DERIVED = ("is_text", "urls", "domains", "emojis", "text_clean", "tokens", "tokens_content", "language",
            "n_words", "n_chars", "n_emojis")


def anonymize(df: pd.DataFrame, mask_text: bool = True) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace sender names with ``Participant N`` and (optionally) scrub identifiers in message text.

    MUST be applied to the parsed frame BEFORE ``preprocess`` so that every derived text feature
    (tokens, n-grams, keywords, language) is computed from the scrubbed text and cannot leak a name.
    Text masking covers: participant names mentioned verbatim, phone numbers, e-mail addresses, and URL
    paths/query strings (the domain is kept, since it is useful for link analytics and is not personal).
    It is best-effort: free text can contain other identifying details this cannot find.
    Returns the anonymised copy and the mapping (keep the mapping private)."""
    if any(c in df for c in _DERIVED):
        raise ValueError("anonymize() must run before preprocess(): derived text columns would leak names")
    mapping = participant_mapping(df)
    out = df.copy()
    out["sender"] = out["sender"].map(mapping).where(out["sender"].notna()).astype("object")
    if mask_text:
        names = sorted(mapping, key=len, reverse=True)
        name_rx = re.compile("|".join(re.escape(n) for n in names if len(n) >= 2),
                             re.IGNORECASE) if names else None
        lower_map = {k.lower(): v for k, v in mapping.items()}

        def scrub(m: str) -> str:
            m = _EMAIL_RE.sub("[email]", m)
            m = _URL_RE.sub(lambda x: f"https://{x.group(1)}/[path]", m)
            m = _PHONE_RE.sub("[phone]", m)
            if name_rx:
                m = name_rx.sub(lambda x: lower_map.get(x.group(0).lower(), "[name]"), m)
            return m

        out["message"] = out["message"].map(scrub)
    return out, mapping
