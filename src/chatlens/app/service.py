"""Application service layer: upload -> validated analysis -> exports. No UI code here, so every
privacy/security behaviour is unit-testable.

Privacy behaviour (documented in README):
 * the upload is read into memory and the temporary upload file is deleted immediately;
 * names are anonymised by default; exports omit message text unless the user opts in;
 * working folders are mode 0700 and deleted after ``retention_minutes``;
 * nothing is sent to any external service; log lines carry counts and timings only.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from ..analyze import AnalysisBundle, NLPOptions, analyze_chat
from ..config import AnalysisConfig
from ..exceptions import ChatLensError
from ..ingestion import read_chat
from ..logging_utils import get_logger
from ..nlp.chat_nlp import LabelModel, LanguageRouter, vader_model
from ..report import export_bundle

log = get_logger("chatlens.app")


class UserFacingError(Exception):
    """An error whose message is safe and useful to show to the user."""


@dataclass
class AppSettings:
    max_upload_mb: int = 50
    retention_minutes: int = 30
    work_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "chatlens_work")
    model_dir: Path | None = None
    feedback_path: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "chatlens_feedback.jsonl")

    @classmethod
    def from_env(cls) -> AppSettings:
        e = os.environ
        s = cls()
        s.max_upload_mb = int(e.get("CHATLENS_MAX_UPLOAD_MB", s.max_upload_mb))
        s.retention_minutes = int(e.get("CHATLENS_RETENTION_MINUTES", s.retention_minutes))
        if e.get("CHATLENS_WORK_DIR"):
            s.work_root = Path(e["CHATLENS_WORK_DIR"])
        if e.get("CHATLENS_MODEL_DIR"):
            s.model_dir = Path(e["CHATLENS_MODEL_DIR"])
        if e.get("CHATLENS_FEEDBACK_FILE"):
            s.feedback_path = Path(e["CHATLENS_FEEDBACK_FILE"])
        return s


def cleanup_old_runs(work_root: Path, retention_minutes: int, now: float | None = None) -> int:
    """Delete run folders older than the retention window. Returns how many were removed."""
    now = now or time.time()
    removed = 0
    if not work_root.exists():
        return 0
    for d in work_root.iterdir():
        if d.is_dir() and d.name.startswith("run_") and now - d.stat().st_mtime > retention_minutes * 60:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def purge_all(work_root: Path) -> None:
    shutil.rmtree(work_root, ignore_errors=True)


def load_routers(model_dir: Path | None) -> tuple[LanguageRouter, str]:
    """Sentiment router. English -> VADER (zero-shot). If a trained Roman-Urdu pipeline exists in
    ``model_dir`` it handles messages detected as Hinglish/Roman-Urdu. Returns (router, description)."""
    models = {"en": vader_model()}
    desc = ["English: VADER lexicon (zero-shot)"]
    if model_dir:
        p = Path(model_dir) / "sentiment_roman_urdu.joblib"
        if p.exists():
            try:
                pipe = joblib.load(p)        # trusted, self-produced artifact (hash-verified in the notebook)
                models["hinglish"] = LabelModel("tfidf_logreg(roman-urdu)", pipe)
                desc.append("Hinglish/Roman-Urdu: TF-IDF + logistic regression trained on the Roman Urdu corpus")
            except Exception as exc:        # noqa: BLE001
                log.warning("could not load model: %s", type(exc).__name__)
    return LanguageRouter(models), "; ".join(desc) + ". Other languages are left unscored."


@dataclass
class ServiceResult:
    bundle: AnalysisBundle
    zip_path: Path
    model_description: str
    run_dir: Path


def process_upload(upload: str | Path | bytes, filename: str | None, settings: AppSettings, anonymize: bool = True,
                   with_nlp: bool = True, include_text_in_export: bool = False,
                   progress=None) -> ServiceResult:
    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, desc=msg)

    cfg = AnalysisConfig(max_upload_mb=settings.max_upload_mb)
    cleanup_old_runs(settings.work_root, settings.retention_minutes)
    tick(0.05, "Validating file")
    try:
        path_to_delete = Path(upload) if isinstance(upload, (str, Path)) else None
        ing = read_chat(upload, filename=filename or (path_to_delete.name if path_to_delete else None), config=cfg)
    except ChatLensError as exc:
        raise UserFacingError(str(exc)) from exc
    finally:
        if isinstance(upload, (str, Path)):        # the user's raw chat must not linger in temp storage
            Path(upload).unlink(missing_ok=True)
    if ing.replaced_chars:
        log.warning("decoded with %d replacement characters", ing.replaced_chars)
    tick(0.15, "Parsing and analysing")
    router, desc = load_routers(settings.model_dir) if with_nlp else (None, "NLP disabled")
    opts = NLPOptions(sentiment=router, run_topics=with_nlp, run_keywords=with_nlp, run_summaries=with_nlp)
    try:
        bundle = analyze_chat(ing.text, cfg, anonymize_names=anonymize, nlp=opts, source_name="upload")
    except ChatLensError as exc:
        raise UserFacingError(str(exc)) from exc
    except Exception as exc:        # noqa: BLE001
        log.error("analysis failed: %s", type(exc).__name__)       # type only: never message content
        raise UserFacingError("The file could not be analysed. It may not be a standard WhatsApp export "
                              f"({type(exc).__name__}).") from exc
    if bundle.overview["total_messages"] == 0:
        raise UserFacingError("No participant messages were found (only system notices).")
    tick(0.85, "Building report")
    settings.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = settings.work_root / f"run_{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(mode=0o700)
    export_dir = run_dir / "export"
    export_bundle(bundle, export_dir, include_text=include_text_in_export and not anonymize)
    zip_path = run_dir / "chatlens_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(export_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(export_dir))
    tick(1.0, "Done")
    log.info("analysis complete rows=%d anonymize=%s nlp=%s", len(bundle.df), anonymize, with_nlp)
    return ServiceResult(bundle, zip_path, desc, run_dir)
