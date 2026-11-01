"""Regular expressions and marker lists for WhatsApp exports.

Supported header layouts (all verified by tests):
  Android: ``12/31/23, 9:41 PM - Name: text``  /  ``31/12/2023, 21:41 - Name: text``
  iOS:     ``[31/12/2023, 21:41:05] Name: text`` / ``[12/31/23, 9:41:05 PM] Name: text``
  Separators ``/ . -`` and ISO ``2023-12-31`` dates. Localised system strings and localised
  "<Media omitted>" markers other than English are NOT built in; add them via
  ``AnalysisConfig.extra_media_markers``.
"""
from __future__ import annotations

import re

_DATE = r"(?P<d1>\d{1,4})[/.\-](?P<d2>\d{1,2})[/.\-](?P<d3>\d{1,4})"
_TIME = (r"(?P<h>\d{1,2})[:.](?P<m>\d{2})(?:[:.](?P<s>\d{2}))?"
         r"\s?(?P<ap>[AaPp]\.?\s?[Mm]\.?)?")

ANDROID_RE = re.compile(rf"^{_DATE},?\s+{_TIME}\s+-\s+(?P<rest>.*)$")
IOS_RE = re.compile(rf"^\[{_DATE},?\s+{_TIME}\]\s+(?P<rest>.*)$")

SENDER_SPLIT_RE = re.compile(r"^(?P<sender>[^:]{1,100}?):\s(?P<text>.*)$", re.DOTALL)

# Characters WhatsApp injects around text (bidi marks, narrow no-break space).
INVISIBLE_TRANSLATION = {ord(c): None for c in
                         "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"}
SPACE_TRANSLATION = {0x202F: " ", 0x00A0: " "}

SYSTEM_RE = re.compile(
    r"end-to-end encrypted|created group|created this group|\badded\b|\bremoved\b|"
    r"\bleft\b$|joined using|joined from|changed (this group's|the group|the subject|their phone|"
    r"your phone|the group description)|changed this group|security code|now an admin|"
    r"no longer an admin|missed (voice|video) call|pinned a message|disappearing messages|"
    r"turned (on|off) |was deleted by|you were added|group (icon|description)|"
    r"requested to join|business account",
    re.IGNORECASE,
)

MEDIA_RE = re.compile(
    r"^(<media omitted>|<attached: .*>|(image|video|audio|sticker|gif|document|contact card|"
    r"video note|voice message)\s+omitted|.*\(file attached\)|\S+\.(jpg|jpeg|png|webp|mp4|opus|"
    r"m4a|pdf|vcf|docx?|xlsx?|pptx?)\s*\(?.*attached.*)$",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(r"^location:\s*https?://", re.IGNORECASE)
DELETED_RE = re.compile(r"^(this message was deleted|you deleted this message|"
                        r"the message was deleted)\.?$", re.IGNORECASE)
EDITED_SUFFIX_RE = re.compile(r"\s*<this message was edited>\s*$", re.IGNORECASE)

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)
