"""WhatsApp export parser.

Design rules
------------
* Nothing is silently dropped: every line either becomes part of a message row or is counted in
  :class:`ParseReport` (``unparsed_leading_lines``).
* Timestamps that cannot be interpreted keep the row with ``timestamp = NaT`` and
  ``timestamp_invalid = True``.
* Day/month order is inferred from the data (any value > 12 is decisive; otherwise the order that
  yields fewer chronological regressions wins; ties default to day-first and are reported).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

from ..config import AnalysisConfig
from ..exceptions import ParseError
from . import patterns as P


@dataclass
class ParseReport:
    source_format: str = "unknown"
    total_lines: int = 0
    blank_lines: int = 0
    header_lines: int = 0
    messages: int = 0
    multiline_messages: int = 0
    system_messages: int = 0
    media_messages: int = 0
    deleted_messages: int = 0
    edited_messages: int = 0
    empty_messages: int = 0
    unparsed_leading_lines: int = 0
    invalid_timestamps: int = 0
    duplicate_rows_flagged: int = 0
    date_order: str = "unknown"          # "day-first" | "month-first" | "year-first"
    date_order_confident: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseResult:
    df: pd.DataFrame
    report: ParseReport


@dataclass
class _Raw:
    line_no: int
    d1: int
    d2: int
    d3: int
    h: int
    m: int
    s: int
    ap: str | None
    rest: str
    extra: list[str] = field(default_factory=list)


def _clean(line: str) -> str:
    return line.translate(P.INVISIBLE_TRANSLATION).translate(P.SPACE_TRANSLATION)


def _match_header(line: str):
    for fmt, rx in (("ios", P.IOS_RE), ("android", P.ANDROID_RE)):
        m = rx.match(line)
        if m:
            return fmt, m
    return None, None


def _infer_order(raws: list[_Raw]) -> tuple[str, bool]:
    if any(len(str(r.d1)) == 4 or r.d1 > 31 for r in raws):
        return "year-first", True
    first_gt12 = any(r.d1 > 12 for r in raws)
    second_gt12 = any(r.d2 > 12 for r in raws)
    if first_gt12 and not second_gt12:
        return "day-first", True
    if second_gt12 and not first_gt12:
        return "month-first", True
    if first_gt12 and second_gt12:
        return "day-first", False  # contradictory evidence

    def regressions(day_first: bool) -> int:
        prev, bad = None, 0
        for r in raws:
            d, mth = (r.d1, r.d2) if day_first else (r.d2, r.d1)
            key = (r.d3, mth, d)
            if prev is not None and key < prev:
                bad += 1
            prev = key
        return bad

    a, b = regressions(True), regressions(False)
    if a < b:
        return "day-first", True
    if b < a:
        return "month-first", True
    return "day-first", False


def _to_datetime(r: _Raw, order: str) -> datetime | None:
    try:
        if order == "year-first":
            year, month, day = r.d1, r.d2, r.d3
        elif order == "day-first":
            day, month, year = r.d1, r.d2, r.d3
        else:
            month, day, year = r.d1, r.d2, r.d3
        if year < 100:
            year += 2000
        hour = r.h
        if r.ap:
            ap = r.ap.lower().replace(".", "").replace(" ", "")
            if not 1 <= hour <= 12:
                return None
            hour = (hour % 12) + (12 if ap == "pm" else 0)
        return datetime(year, month, day, hour, r.m, r.s)
    except ValueError:
        return None


def parse_chat(text: str, config: AnalysisConfig | None = None) -> ParseResult:
    """Parse the text of a WhatsApp export into a message DataFrame plus a ParseReport."""
    cfg = config or AnalysisConfig()
    rep = ParseReport()
    raws: list[_Raw] = []
    fmt_counts = {"android": 0, "ios": 0}
    current: _Raw | None = None

    for line_no, original in enumerate(text.splitlines(), start=1):
        rep.total_lines += 1
        line = _clean(original)
        fmt, m = _match_header(line)
        if m:
            fmt_counts[fmt] += 1
            rep.header_lines += 1
            g = m.groupdict()
            current = _Raw(line_no, int(g["d1"]), int(g["d2"]), int(g["d3"]), int(g["h"]),
                           int(g["m"]), int(g["s"] or 0), g["ap"], g["rest"])
            raws.append(current)
        elif current is not None:
            if line.strip() == "":
                rep.blank_lines += 1
            current.extra.append(line)
        elif line.strip():
            rep.unparsed_leading_lines += 1
        else:
            rep.blank_lines += 1

    if not raws:
        raise ParseError("No WhatsApp message headers were found. Is this an exported chat .txt?")
    if rep.header_lines < 3 and rep.total_lines > 50:
        rep.warnings.append("Very few message headers relative to file size; format may differ.")

    rep.source_format = max(fmt_counts, key=fmt_counts.get)
    if all(v > 0 for v in fmt_counts.values()):
        rep.warnings.append("Both Android and iOS header styles detected in one file.")

    order, confident = _infer_order(raws)
    rep.date_order, rep.date_order_confident = order, confident
    if not confident:
        rep.warnings.append(
            "Day/month order is ambiguous; assumed day-first. Verify dates or re-export.")

    media_extra = tuple(s.lower() for s in cfg.extra_media_markers)
    rows = []
    for r in raws:
        body = r.rest
        if r.extra:
            body = body + "\n" + "\n".join(r.extra)
            rep.multiline_messages += 1
        body = body.rstrip("\n")
        sender, msg, is_system = None, body, False
        split = P.SENDER_SPLIT_RE.match(body)
        if split and not P.SYSTEM_RE.search(split.group("sender")):
            sender, msg = split.group("sender").strip(), split.group("text")
        else:
            is_system = True
        # a header with only "Name:" and nothing after
        if sender is None and body.rstrip().endswith(":") and "\n" not in body \
                and not P.SYSTEM_RE.search(body):
            sender, msg, is_system = body.rstrip()[:-1].strip(), "", False

        stripped = msg.strip()
        edited = bool(P.EDITED_SUFFIX_RE.search(stripped))
        if edited:
            stripped = P.EDITED_SUFFIX_RE.sub("", stripped)
        is_deleted = bool(P.DELETED_RE.match(stripped)) and not is_system
        is_media = (not is_system) and (
            bool(P.MEDIA_RE.match(stripped)) or any(k in stripped.lower() for k in media_extra
                                                    if k))
        is_location = (not is_system) and bool(P.LOCATION_RE.match(stripped))
        ts = _to_datetime(r, order)
        rows.append({
            "line_no": r.line_no, "timestamp": ts, "sender": sender, "message": stripped,
            "is_system": is_system, "is_media": is_media, "is_deleted": is_deleted,
            "is_edited": edited, "is_location": is_location,
            "timestamp_invalid": ts is None,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_empty"] = (~df["is_system"]) & (~df["is_media"]) & (~df["is_deleted"]) \
        & (df["message"].str.strip() == "")
    # Flag, never remove, repeated (timestamp, sender, message) rows. Legitimate repeats exist.
    df["is_duplicate"] = df.duplicated(["timestamp", "sender", "message"], keep="first") \
        & df["timestamp"].notna() & (~df["is_system"])
    df.insert(0, "msg_id", range(len(df)))

    rep.messages = len(df)
    rep.system_messages = int(df["is_system"].sum())
    rep.media_messages = int(df["is_media"].sum())
    rep.deleted_messages = int(df["is_deleted"].sum())
    rep.edited_messages = int(df["is_edited"].sum())
    rep.empty_messages = int(df["is_empty"].sum())
    rep.invalid_timestamps = int(df["timestamp_invalid"].sum())
    rep.duplicate_rows_flagged = int(df["is_duplicate"].sum())
    if rep.unparsed_leading_lines:
        rep.warnings.append(
            f"{rep.unparsed_leading_lines} non-empty line(s) before the first header were "
            "not attributable to any message and are reported here, not parsed.")
    if rep.invalid_timestamps:
        rep.warnings.append(
            f"{rep.invalid_timestamps} message(s) have an invalid timestamp; kept but excluded "
            "from time-based analytics.")
    return ParseResult(df=df, report=rep)
