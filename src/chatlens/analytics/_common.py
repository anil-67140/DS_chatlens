from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def user_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Messages written by participants (system notices excluded), with valid timestamps."""
    return df[(~df["is_system"]) & df["timestamp"].notna()]


def text_messages(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_text"] & (~df["is_system"])] if "is_text" in df else user_messages(df)
