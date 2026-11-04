import json

import numpy as np
import pandas as pd
import pytest

from chatlens import parse_chat, preprocess, validate
from chatlens.analytics import behavior, participants, temporal
from chatlens.monitoring import FeedbackStore, nlp_output_checks, psi, resource_monitor
from chatlens.synthetic import make_synthetic_chat
from chatlens.tracking import Tracker, dataset_fingerprint


@pytest.fixture(scope="module")
def synth():
    text, truth = make_synthetic_chat(n_messages=4000, seed=7)
    res = parse_chat(text)
    return res, preprocess(res.df), truth


def test_synthetic_is_deterministic_and_flagged_fabricated():
    a, ta = make_synthetic_chat(500, seed=1)
    b, _ = make_synthetic_chat(500, seed=1)
    c, _ = make_synthetic_chat(500, seed=2)
    assert a == b and a != c and ta.fabricated and "FABRICATED" in ta.notes


def test_synthetic_parses_cleanly_in_all_formats():
    for fmt in ("android_24h", "android_12h", "ios"):
        text, _ = make_synthetic_chat(600, seed=3, fmt=fmt)
        res = parse_chat(text)
        assert res.report.unparsed_leading_lines == 0 and res.report.invalid_timestamps == 0, fmt
        assert res.report.messages == res.report.header_lines
        assert validate(preprocess(res.df), res.report).ok, fmt


def test_pipeline_recovers_generator_parameters(synth):
    res, df, truth = synth
    assert res.report.system_messages >= 2 and res.report.multiline_messages > 50
    assert res.report.media_messages > 100
    tbl = participants.participant_table(df)
    assert set(tbl.index) == set(truth.participants)
    assert tbl.index[0] == "Ali"                                   # largest target share is recovered first
    assert tbl["messages"].is_monotonic_decreasing or tbl.loc["Ali", "messages"] > tbl.loc["Usman", "messages"]
    top = temporal.top_hours(df, 3)
    assert any(abs(h - truth.peak_hour) <= 2 for h in top)         # evening peak (21h) is found
    rs = behavior.response_summary(df)
    assert 1 <= rs["overall"]["median_min"] <= 4                   # generator median gap = 2 min
    assert df["language"].isin(["en", "hinglish", "ur", "hi", "mixed", "und"]).all()
    assert {"en", "hinglish", "ur", "hi"} <= set(df["language"])   # all four planted languages are detected


def test_psi_known_values():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == pytest.approx(0, abs=1e-9)
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}) > 0.25
    assert 0.1 < psi({"a": 0.5, "b": 0.5}, {"a": 0.7, "b": 0.3}) < 0.25  # = 0.17 by hand


def test_nlp_output_checks_alerts():
    ref = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    ok = nlp_output_checks(["positive", "negative", "neutral"] * 20, np.full(60, 0.8), ref, 0.75, 100)
    assert ok["alerts"] == [] and ok["coverage"] == 0.6
    bad = nlp_output_checks(["positive"] * 95 + ["negative"] * 5, np.full(100, 0.3), ref, 0.75)
    joined = " ".join(bad["alerts"])
    assert "major shift" in joined and "90%" in joined and "confidence" in joined
    assert any("fewer than 30" in a for a in nlp_output_checks(["positive"] * 5, None, ref)["alerts"])


def test_resource_monitor_records():
    log = {}
    with resource_monitor(log, "stage"):
        sum(range(10**5))
    assert log["stage"]["wall_s"] >= 0 and "rss_mb" in log["stage"]


def test_feedback_store_validates_truncates_and_protects(tmp_path):
    fs = FeedbackStore(tmp_path / "fb" / "f.jsonl")
    fs.add("topics", 4, "useful")
    long = fs.add("sentiment", 2, "x" * 900)
    leaked = fs.add("overview", 3, "he said: meet me at the old station tonight ok",
                    forbidden_substrings=["meet me at the old station tonight"])
    assert len(long["comment"]) == 500 and "removed" in leaked["comment"]
    with pytest.raises(ValueError):
        fs.add("x", 9)
    assert len(fs.read()) == 3 and fs.summary()["topics"]["mean_rating"] == 4
    assert oct((tmp_path / "fb" / "f.jsonl").stat().st_mode)[-3:] == "600"


def test_dataset_fingerprint_is_order_independent_and_content_sensitive():
    a = pd.DataFrame({"text": ["x", "y"], "label": ["p", "n"]})
    assert dataset_fingerprint(a) == dataset_fingerprint(a.iloc[::-1])
    b = a.copy()
    b.loc[0, "label"] = "n"
    assert dataset_fingerprint(a) != dataset_fingerprint(b)


@pytest.mark.parametrize("use_mlflow", [True, False])
def test_tracker_logs_both_backends(tmp_path, use_mlflow):
    if use_mlflow:
        pytest.importorskip("mlflow")
    t = Tracker(tmp_path, "unit", use_mlflow=use_mlflow)
    assert t.backend == ("mlflow" if use_mlflow else "jsonl")
    art = tmp_path / "r.json"
    art.write_text("{}")
    t.log_run("run1", {"k": 5, "cfg": {"a": 1}}, {"macro_f1": 0.5, "bad": float("nan")}, [art])
    rec = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert rec["metrics"] == {"macro_f1": 0.5} and rec["env"]["chatlens"]
    if use_mlflow:
        import mlflow
        runs = mlflow.search_runs(experiment_names=["unit"])
        assert len(runs) == 1 and runs.iloc[0]["metrics.macro_f1"] == 0.5
        assert (tmp_path / "mlflow.db").exists()


def test_tracker_sync_copies_local_db_to_drive(tmp_path):
    pytest.importorskip("mlflow")
    drive, local = tmp_path / "drive", tmp_path / "local"
    t = Tracker(drive, "sync", local_dir=local)
    t.log_run("r", {"a": 1}, {"m": 1.0})
    assert not (drive / "mlflow.db").exists()
    assert t.sync() == drive / "mlflow.db" and (drive / "mlflow.db").stat().st_size > 0
