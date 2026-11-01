"""Data-quality validation of a parsed chat. Reports; never mutates or drops data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .parsing.parser import ParseReport

WHATSAPP_LAUNCH = pd.Timestamp("2009-01-01")


@dataclass
class QualityCheck:
    name: str
    passed: bool
    severity: str        # "error" | "warning" | "info"
    detail: str
    count: int = 0


@dataclass
class QualityReport:
    checks: list[QualityCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any((not c.passed) and c.severity == "error" for c in self.checks)

    def failed(self) -> list[QualityCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}


def infer_chat_type(df: pd.DataFrame) -> str:
    senders = df.loc[~df["is_system"], "sender"].nunique()
    sys_text = df.loc[df["is_system"], "message"].str.lower()
    group_hint = sys_text.str.contains("created group|added|joined using|group", regex=True).any()
    if senders > 2 or group_hint:
        return "group"
    return "individual"


def validate(df: pd.DataFrame, report: ParseReport | None = None,
             now: pd.Timestamp | None = None) -> QualityReport:
    now = now or pd.Timestamp.now()
    q = QualityReport()
    add = lambda *a, **k: q.checks.append(QualityCheck(*a, **k))  # noqa: E731

    n = len(df)
    add("non_empty", n > 0, "error", f"{n} message rows parsed", n)

    ts = df["timestamp"]
    bad_ts = int(ts.isna().sum())
    add("timestamps_valid", bad_ts == 0, "warning",
        f"{bad_ts} rows have unparseable timestamps (kept, excluded from time analytics)", bad_ts)

    valid = ts.dropna()
    out_of_range = int(((valid < WHATSAPP_LAUNCH) | (valid > now + pd.Timedelta(days=1))).sum())
    add("timestamps_plausible", out_of_range == 0, "error",
        f"{out_of_range} timestamps outside 2009-01-01..now", out_of_range)

    regress = int((valid.diff() < pd.Timedelta(0)).sum())
    add("chronological_order", regress == 0, "warning",
        f"{regress} timestamp regressions (exports are normally chronological; may indicate wrong "
        "day/month order or a merged file)", regress)

    user = df[~df["is_system"]]
    senders = user["sender"].dropna()
    blank = int((senders.str.strip() == "").sum())
    add("sender_names_non_blank", blank == 0, "error", f"{blank} blank sender names", blank)
    n_senders = senders.nunique()
    add("sender_count_plausible", n_senders <= 1024, "error",
        f"{n_senders} unique senders (WhatsApp groups are capped at 1024)", n_senders)
    long_names = int((senders.str.len() > 60).sum())
    add("sender_name_length", long_names == 0, "warning",
        f"{long_names} sender names > 60 chars (possible mis-split of a message)", long_names)
    # A sender that appears exactly once among many is a typical mis-parse signature.
    singletons = int((senders.value_counts() == 1).sum())
    add("singleton_senders", not (n_senders > 5 and singletons / max(n_senders, 1) > 0.5),
        "warning", f"{singletons} senders appear exactly once", singletons)

    dups = int(df["is_duplicate"].sum()) if "is_duplicate" in df else 0
    add("duplicates", dups == 0, "info",
        f"{dups} rows share timestamp+sender+text with an earlier row (flagged, not removed)", dups)

    empty = int(df["is_empty"].sum()) if "is_empty" in df else 0
    add("empty_messages", empty == 0, "info", f"{empty} messages have no text", empty)

    if report is not None:
        add("date_order_confident", report.date_order_confident, "warning",
            f"date order assumed {report.date_order}")
        add("no_unparsed_lines", report.unparsed_leading_lines == 0, "warning",
            f"{report.unparsed_leading_lines} lines before first header",
            report.unparsed_leading_lines)
    return q
