"""Anonymisation and masking helpers. Deterministic (by first appearance), no external calls."""
from __future__ import annotations

import re

import pandas as pd

_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s\-().]{7,}\d(?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def participant_mapping(df: pd.DataFrame) -> dict[str, str]:
    order = df.loc[~df["is_system"], "sender"].dropna().drop_duplicates().tolist()
    return {name: f"Participant {i + 1}" for i, name in enumerate(order)}


def anonymize(df: pd.DataFrame, mask_text: bool = True) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace sender names with ``Participant N`` and (optionally) scrub identifiers in text.

    Text masking covers: participant names mentioned verbatim, phone numbers, e-mail addresses
    and URLs. It is best-effort: free-text can contain other identifying details this cannot find.
    Returns the anonymised copy and the mapping (keep the mapping private).
    """
    mapping = participant_mapping(df)
    out = df.copy()
    out["sender"] = out["sender"].map(mapping).where(out["sender"].notna())
    if mask_text:
        names = sorted(mapping, key=len, reverse=True)
        name_rx = re.compile("|".join(re.escape(n) for n in names if len(n) >= 2),
                             re.IGNORECASE) if names else None

        def scrub(m: str) -> str:
            m = _EMAIL_RE.sub("[email]", m)
            m = _PHONE_RE.sub("[phone]", m)
            m = re.sub(r"(?:https?://|www\.)\S+", "[url]", m, flags=re.IGNORECASE)
            if name_rx:
                m = name_rx.sub(lambda x: mapping.get(
                    next((k for k in mapping if k.lower() == x.group(0).lower()), ""), "[name]"), m)
            return m

        out["message"] = out["message"].map(scrub)
        for col in ("text_clean",):
            if col in out:
                out[col] = out[col].map(scrub)
        for col in ("tokens", "tokens_content", "urls", "domains"):
            if col in out:
                out[col] = [[] for _ in range(len(out))]
    return out, mapping
