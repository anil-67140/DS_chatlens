import os
import time
import zipfile

import pytest

from chatlens.app.service import (
    AppSettings,
    UserFacingError,
    cleanup_old_runs,
    load_routers,
    process_upload,
    purge_all,
)
from chatlens.synthetic import make_synthetic_chat


@pytest.fixture()
def settings(tmp_path):
    return AppSettings(work_root=tmp_path / "work", retention_minutes=30, max_upload_mb=5,
                       feedback_path=tmp_path / "fb.jsonl")


@pytest.fixture(scope="module")
def chat_text():
    return make_synthetic_chat(1200, seed=21)[0]


def test_happy_path_anonymises_and_zip_has_no_text(settings, tmp_path, chat_text):
    f = tmp_path / "chat.txt"
    f.write_text(chat_text, encoding="utf-8")
    steps = []
    r = process_upload(f, "chat.txt", settings, progress=lambda frac, desc: steps.append(desc))
    assert not f.exists()                                          # raw upload deleted immediately
    assert steps[0] == "Validating file" and steps[-1] == "Done"
    assert oct(r.run_dir.stat().st_mode)[-3:] == "700"
    with zipfile.ZipFile(r.zip_path) as z:
        names = z.namelist()
        assert "report.html" in names and "summary.json" in names
        blob = b"".join(z.read(n) for n in names).decode("utf-8", "ignore")
        row_level = b"".join(z.read(n) for n in names if n.endswith(".csv") and "processed" in n).decode()
    for real in ("Ali", "Sara", "Bilal", "Ayesha", "Usman"):
        assert f"{real}:" not in blob and f'"{real}"' not in blob and f"<td>{real}</td>" not in blob
    assert "Participant 1" in blob
    # Row-level data never carries message text. (Aggregates - n-grams, RAKE phrases - may contain
    # repeated message fragments by design; the report is for the data owner. See README.)
    assert "good morning everyone" not in row_level and "message" not in row_level.splitlines()[0].split(",")


def test_rejects_bad_inputs_with_friendly_messages(settings):
    with pytest.raises(UserFacingError, match="empty"):
        process_upload(b"", "a.txt", settings)
    with pytest.raises(UserFacingError, match=r"\.txt"):
        process_upload(b"hello", "evil.exe", settings)
    with pytest.raises(UserFacingError, match="WhatsApp"):
        process_upload(b"just some text\nno headers", "a.txt", settings)
    with pytest.raises(UserFacingError, match="limit"):
        process_upload(b"x" * (6 * 1024 * 1024), "a.txt", settings)
    with pytest.raises(UserFacingError, match="system notices"):
        process_upload(b"[01/01/2024, 10:00:00] Messages and calls are end-to-end encrypted.\n", "a.txt", settings)


def test_failed_upload_is_still_deleted(settings, tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("not a chat")
    with pytest.raises(UserFacingError):
        process_upload(f, "bad.txt", settings)
    assert not f.exists()


def test_retention_cleanup(settings, chat_text):
    r = process_upload(chat_text.encode(), "a.txt", settings, with_nlp=False)
    assert r.run_dir.exists()
    assert cleanup_old_runs(settings.work_root, 30) == 0
    old = time.time() - 31 * 60
    os.utime(r.run_dir, (old, old))
    assert cleanup_old_runs(settings.work_root, 30) == 1 and not r.run_dir.exists()
    purge_all(settings.work_root)
    assert not settings.work_root.exists()


def test_text_export_only_when_not_anonymised_and_requested(settings, chat_text):
    r = process_upload(chat_text.encode(), "a.txt", settings, anonymize=False, include_text_in_export=True,
                       with_nlp=False)
    with zipfile.ZipFile(r.zip_path) as z:
        assert "message" in z.read("processed_messages.csv").decode().splitlines()[0]
    r2 = process_upload(chat_text.encode(), "a.txt", settings, anonymize=True, include_text_in_export=True,
                        with_nlp=False)           # anonymised: text export is refused even if requested
    with zipfile.ZipFile(r2.zip_path) as z:
        assert "message" not in z.read("processed_messages.csv").decode().splitlines()[0].split(",")


def test_router_uses_trained_model_when_present(tmp_path):
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("v", TfidfVectorizer()), ("c", LogisticRegression())]).fit(
        ["acha bohot", "bura bohot", "theek hai"] * 5, ["positive", "negative", "neutral"] * 5)
    joblib.dump(pipe, tmp_path / "sentiment_roman_urdu.joblib")
    router, desc = load_routers(tmp_path)
    assert set(router.models) == {"en", "hinglish"} and "Roman-Urdu" in desc
    assert set(load_routers(None)[0].models) == {"en"}
    (tmp_path / "sentiment_roman_urdu.joblib").write_bytes(b"corrupt")
    assert set(load_routers(tmp_path)[0].models) == {"en"}          # corrupt artifact degrades gracefully


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATLENS_MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("CHATLENS_WORK_DIR", str(tmp_path / "w"))
    s = AppSettings.from_env()
    assert s.max_upload_mb == 7 and s.work_root == tmp_path / "w"
