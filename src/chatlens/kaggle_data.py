"""Kaggle dataset download (resumable, atomic) and loading/cleaning of labelled text corpora.

Datasets used (verified listings; licences are those stated by the dataset authors and must be
re-checked on the Kaggle page before redistribution):
  * Roman Urdu sentiment  - ``smat26/sentiment-analysis-on-roman-urdu``; 20k+ manually tagged
    e-commerce/Facebook/Twitter sentences, Positive/Neutral/Negative. Mirror repo is GPL-3.0.
  * Emotions for NLP      - ``praveengovi/emotions-dataset-for-nlp``; English tweets, 6 emotions,
    official train/val/test split.
  * GoEmotions            - ``shivamb/go-emotions-google-emotions-dataset``; ~58k Reddit comments
    (used as an unlabeled informal-text corpus for topic/keyword work).
  * Sentiment140          - ``kazanova/sentiment140``; 1.6M tweets, labels from emoticons
    (distantly supervised: noisy by construction).
None of these is a real WhatsApp corpus. WhatsApp chats are private; public Kaggle "chat" files
are either synthetic or personal data of unknown consent, and are deliberately not used.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .checkpoint import atomic_write_json
from .logging_utils import get_logger

log = get_logger()

CANDIDATES = {
    "roman_urdu": ["smat26/sentiment-analysis-on-roman-urdu", "smat26/roman-urdu-dataset"],
    "emotions": ["praveengovi/emotions-dataset-for-nlp"],
    "goemotions": ["shivamb/go-emotions-google-emotions-dataset"],
    "sentiment140": ["kazanova/sentiment140"],
}


# ------------------------------------------------------------------------------ authentication
def configure_kaggle_auth(token: str | None = None, username: str | None = None,
                          key: str | None = None) -> str:
    """Set Kaggle credentials in the process environment (never printed or logged).

    Supports the current access-token flow (``KAGGLE_API_TOKEN``) and the legacy
    ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` pair. Returns the method used."""
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        return "access_token"
    if username and key:
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = username, key
        return "legacy_key"
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "access_token(env)"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy_key(env)"
    if (Path.home() / ".kaggle" / "access_token").exists() \
            or (Path.home() / ".kaggle" / "kaggle.json").exists():
        return "file"
    raise RuntimeError("No Kaggle credentials found. Set KAGGLE_API_TOKEN (Colab Secret) or "
                       "KAGGLE_USERNAME + KAGGLE_KEY.")


def _redact(text: str) -> str:
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_KEY"):
        v = os.environ.get(var)
        if v:
            text = text.replace(v, "***")
    return text


@dataclass
class DownloadResult:
    key: str
    slug: str
    path: str
    files: list[str]
    skipped: bool
    tried: list[str] = field(default_factory=list)


def _list_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and not p.name.startswith("."))


def download_dataset(key: str, dest_root: str | Path, slugs: list[str] | None = None,
                     timeout: int = 1800, runner=subprocess.run) -> DownloadResult:
    """Download the first working slug for ``key`` into ``dest_root/key``.

    Resumable and atomic: files are downloaded into ``<key>.partial`` and the directory is
    renamed only when the CLI succeeded, so an interrupted download never looks complete.
    A ``_COMPLETE.json`` marker lists the files; if they are all still present the download is
    skipped."""
    dest_root = Path(dest_root)
    dest = dest_root / key
    marker = dest / "_COMPLETE.json"
    if marker.exists():
        import json
        info = json.loads(marker.read_text())
        if all((dest / f).exists() for f in info["files"]):
            return DownloadResult(key, info["slug"], str(dest), info["files"], True)
    slugs = slugs or CANDIDATES[key]
    tried: list[str] = []
    for slug in slugs:
        part = dest_root / f"{key}.partial"
        shutil.rmtree(part, ignore_errors=True)
        part.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(part), "--unzip", "-q"]
        try:
            proc = runner(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            tried.append(f"{slug}: timeout")
            continue
        if proc.returncode != 0 or not _list_files(part):
            tried.append(f"{slug}: {_redact((proc.stderr or proc.stdout or '').strip())[:200]}")
            continue
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(part, dest)
        files = _list_files(dest)
        atomic_write_json(dest / "_COMPLETE.json", {"slug": slug, "files": files})
        log.info("downloaded %s (%d files)", slug, len(files))
        return DownloadResult(key, slug, str(dest), files, False, tried)
    raise RuntimeError(f"Could not download '{key}'. Tried: {tried}. Check credentials, that you "
                       "accepted the dataset terms on kaggle.com, or search with "
                       "`kaggle datasets list -s <term>`.")


# ------------------------------------------------------------------------------------ cleaning
_LABEL_FIXES = {"neative": "negative", "negatve": "negative", "positve": "positive",
                "netural": "neutral", "nuetral": "neutral"}


def text_key(s: str) -> str:
    """Normalised key for duplicate detection: case/Unicode/punctuation/whitespace-insensitive."""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"https?://\S+|@\w+", " ", s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class DataQualityReport:
    n_raw: int = 0
    dropped_missing_text: int = 0
    dropped_missing_label: int = 0
    labels_repaired: int = 0
    dropped_exact_dup: int = 0
    dropped_conflicting: int = 0
    n_clean: int = 0
    class_counts: dict = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_labeled(df: pd.DataFrame, text_col: str = "text", label_col: str = "label",
                  min_chars: int = 2) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean a labelled corpus and account for every dropped row.

    Duplicates are collapsed on the normalised text key *before* any split, otherwise the same
    sentence can land in train and test and inflate every metric (data leakage). Texts that
    occur with different labels are removed entirely (label noise we cannot adjudicate)."""
    rep = DataQualityReport(n_raw=len(df))
    d = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
    miss_t = d["text"].isna() | (d["text"].astype(str).str.strip().str.len() < min_chars)
    rep.dropped_missing_text = int(miss_t.sum())
    d = d[~miss_t]
    miss_l = d["label"].isna() | (d["label"].astype(str).str.strip() == "")
    rep.dropped_missing_label = int(miss_l.sum())
    d = d[~miss_l].copy()
    d["text"] = d["text"].astype(str).str.strip()
    lab = d["label"].astype(str).str.strip().str.casefold()
    fixed = lab.map(lambda x: _LABEL_FIXES.get(x, x))
    rep.labels_repaired = int((fixed != lab).sum())
    d["label"] = fixed
    d["_key"] = d["text"].map(text_key)
    d = d[d["_key"] != ""]
    nlab = d.groupby("_key")["label"].transform("nunique")
    rep.dropped_conflicting = int((nlab > 1).sum())
    d = d[nlab == 1]
    before = len(d)
    d = d.drop_duplicates("_key", keep="first")
    rep.dropped_exact_dup = before - len(d)
    d = d.drop(columns="_key").reset_index(drop=True)
    rep.n_clean = len(d)
    vc = d["label"].value_counts()
    rep.class_counts = vc.to_dict()
    rep.imbalance_ratio = float(vc.max() / vc.min()) if len(vc) > 1 else 1.0
    if rep.imbalance_ratio > 3:
        rep.notes.append("Class imbalance > 3:1: accuracy is misleading; use macro-F1.")
    if len(vc) < 2:
        rep.notes.append("Only one class present.")
    return d, rep


def split_dataset(df: pd.DataFrame, seed: int = 42, val_size: float = 0.1,
                  test_size: float = 0.2) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split on de-duplicated data (call clean_labeled first)."""
    from sklearn.model_selection import train_test_split
    tr_val, te = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel = val_size / (1 - test_size)
    tr, va = train_test_split(tr_val, test_size=rel, stratify=tr_val["label"], random_state=seed)
    out = {"train": tr.reset_index(drop=True), "val": va.reset_index(drop=True),
           "test": te.reset_index(drop=True)}
    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> dict:
    """Raise if any normalised text appears in more than one split."""
    keys = {n: set(d["text"].map(text_key)) for n, d in splits.items()}
    names = list(keys)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = len(keys[a] & keys[b])
            overlap[f"{a}&{b}"] = c
    if any(overlap.values()):
        raise ValueError(f"Train/val/test leakage detected: {overlap}")
    return overlap


# -------------------------------------------------------------------------------------- loaders
def _find(root: Path, patterns: list[str]) -> Path:
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"None of {patterns} found under {root}")


def load_roman_urdu(root: str | Path) -> pd.DataFrame:
    f = _find(Path(root), ["Roman Urdu DataSet.csv", "*Roman*Urdu*.csv", "*.csv"])
    df = pd.read_csv(f, header=None, on_bad_lines="skip", encoding="utf-8",
                     encoding_errors="replace")
    df = df.iloc[:, :2]
    df.columns = ["text", "label"]
    df["source"] = "roman_urdu"
    return df


def load_emotions(root: str | Path) -> dict[str, pd.DataFrame]:
    """Official splits of praveengovi/emotions-dataset-for-nlp ('text;label' per line)."""
    root = Path(root)
    out = {}
    for split, names in (("train", ["train.txt"]), ("val", ["val.txt", "validation.txt"]),
                         ("test", ["test.txt"])):
        f = _find(root, names)
        rows = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if ";" in line:
                t, lab = line.rsplit(";", 1)
                rows.append((t.strip(), lab.strip()))
        out[split] = pd.DataFrame(rows, columns=["text", "label"])
    return out


def load_goemotions_text(root: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(root).rglob("*.csv")):
        try:
            head = pd.read_csv(f, nrows=5)
        except Exception:
            continue
        if "text" in head.columns:
            frames.append(pd.read_csv(f, usecols=["text"]))
    if not frames:
        raise FileNotFoundError(f"No CSV with a 'text' column under {root}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("text")
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    df["source"] = "goemotions"
    return df.reset_index(drop=True)


def load_sentiment140(root: str | Path, max_rows: int | None = 100_000,
                      seed: int = 42) -> pd.DataFrame:
    f = _find(Path(root), ["*sentiment140*.csv", "training*.csv", "*.csv"])
    df = pd.read_csv(f, header=None, encoding="latin-1",
                     names=["target", "id", "date", "flag", "user", "text"],
                     usecols=["target", "text"])
    df["label"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})
    df = df.dropna(subset=["label"])
    if max_rows and len(df) > max_rows:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(max_rows // df["label"].nunique(), random_state=seed))
    df = df[["text", "label"]].reset_index(drop=True)
    df["source"] = "sentiment140"
    return df
