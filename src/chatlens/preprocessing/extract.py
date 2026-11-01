"""Feature extraction: URLs, domains, emoji."""
from __future__ import annotations

from urllib.parse import urlparse

import emoji

from ..parsing.patterns import URL_RE


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]


def domain_of(url: str) -> str:
    u = url if "//" in url else f"//{url}"
    host = (urlparse(u).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def extract_emojis(text: str) -> list[str]:
    return [e["emoji"] for e in emoji.emoji_list(text)]
