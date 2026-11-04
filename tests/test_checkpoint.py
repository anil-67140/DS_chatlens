import json
import os
import subprocess

import pandas as pd
import pytest

from chatlens import checkpoint as ck
from chatlens.checkpoint import CheckpointStore


def make_stage(store, counter, content="x", name="s1", params=None):
    def fn():
        counter.append(1)
        store.save_json("out/a.json", {"v": content})
    return store.run(name, fn, outputs=["out/a.json"], params=params)


def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "f.bin"
    ck.atomic_write_bytes(target, b"old")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk gone")))
    with pytest.raises(OSError):
        ck.atomic_write_bytes(target, b"new-partial")
    assert target.read_bytes() == b"old"                 # old content intact
    assert not list(tmp_path.glob("*.tmp-*"))            # temp file cleaned


def test_stage_runs_then_skips_when_verified(tmp_path):
    s, calls = CheckpointStore(tmp_path), []
    r1 = make_stage(s, calls)
    r2 = make_stage(s, calls)
    assert not r1.skipped and r2.skipped and len(calls) == 1


def test_resume_across_new_store_instance(tmp_path):
    calls = []
    make_stage(CheckpointStore(tmp_path), calls)
    r = make_stage(CheckpointStore(tmp_path), calls)     # simulates a fresh Colab runtime
    assert r.skipped and len(calls) == 1


def test_changed_params_rerun(tmp_path):
    s, calls = CheckpointStore(tmp_path), []
    make_stage(s, calls, params={"k": 5})
    r = make_stage(s, calls, params={"k": 6})
    assert not r.skipped and "parameters changed" in s.verify("s1", {"k": 5})[1]
    assert len(calls) == 2


def test_corrupted_or_missing_artifact_triggers_rerun(tmp_path):
    s, calls = CheckpointStore(tmp_path), []
    make_stage(s, calls)
    (tmp_path / "out/a.json").write_text('{"v": "tampered"}')
    ok, why = s.verify("s1")
    assert not ok and "checksum" in why
    assert not make_stage(s, calls).skipped
    (tmp_path / "out/a.json").unlink()
    assert "missing" in s.verify("s1")[1]
    assert not make_stage(s, calls).skipped and len(calls) == 3


def test_interrupted_stage_is_marked_and_rerun(tmp_path):
    s = CheckpointStore(tmp_path)

    def dies():
        s.save_json("out/a.json", {"partial": True})
        raise KeyboardInterrupt                           # user pressed stop / runtime died

    with pytest.raises(KeyboardInterrupt):
        s.run("s1", dies, outputs=["out/a.json"])
    assert s.stages["s1"]["status"] == "failed"
    calls = []
    assert not make_stage(CheckpointStore(tmp_path), calls).skipped and calls


def test_crash_leaving_running_status_is_not_trusted(tmp_path):
    s = CheckpointStore(tmp_path)
    s.stages["s1"] = {"status": "running", "artifacts": {}}
    s._save()
    ok, why = CheckpointStore(tmp_path).verify("s1")
    assert not ok and "did not finish" in why


def test_undeclared_output_is_an_error(tmp_path):
    s = CheckpointStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        s.run("s1", lambda: None, outputs=["never.json"])
    assert s.stages["s1"]["status"] == "failed"


def test_dependency_must_be_valid(tmp_path):
    s = CheckpointStore(tmp_path)
    with pytest.raises(RuntimeError, match="depends on"):
        s.run("s2", lambda: None, outputs=[], depends_on=["s1"])


def test_corrupt_manifest_is_preserved_not_fatal(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    s = CheckpointStore(tmp_path)
    assert s.stages == {} and (tmp_path / "manifest.corrupt.json").exists()


def test_orphaned_tmp_files_cleaned(tmp_path):
    (tmp_path / "x.json.tmp-123").write_text("junk")
    CheckpointStore(tmp_path)
    assert not list(tmp_path.glob("*.tmp-*"))


def test_path_traversal_blocked(tmp_path):
    with pytest.raises(ValueError):
        CheckpointStore(tmp_path).path("../escape.txt")


def test_df_roundtrip_with_list_columns(tmp_path):
    s = CheckpointStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "toks": [["x", "y"], []]})
    s.save_df("d.pkl", df)
    pd.testing.assert_frame_equal(s.load_df("d.pkl"), df)


def test_manifest_is_valid_json_with_hashes(tmp_path):
    s, calls = CheckpointStore(tmp_path), []
    make_stage(s, calls)
    m = json.loads((tmp_path / "manifest.json").read_text())
    a = m["stages"]["s1"]["artifacts"]["out/a.json"]
    assert len(a["sha256"]) == 64 and a["bytes"] > 0 and m["stages"]["s1"]["status"] == "complete"
    assert s.summary()[0]["stage"] == "s1"


# --------------------------------------------------------------------------- kaggle download
class FakeRunner:
    """Stands in for the kaggle CLI. ``plan`` maps slug -> ('ok'|'fail'|'timeout'|'empty')."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, cmd, **kw):
        slug, out = cmd[cmd.index("-d") + 1], cmd[cmd.index("-p") + 1]
        self.calls.append(slug)
        mode = self.plan[slug]
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if mode == "ok":
            (open(os.path.join(out, "data.csv"), "w")).write("text,label\nhi,positive\n")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if mode == "empty":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "403 Forbidden token=SECRET123")


def test_download_falls_back_to_next_slug_and_redacts_secrets(tmp_path, monkeypatch):
    from chatlens.kaggle_data import download_dataset
    monkeypatch.setenv("KAGGLE_API_TOKEN", "SECRET123")
    r = FakeRunner({"a/bad": "fail", "b/slow": "timeout", "c/empty": "empty", "d/good": "ok"})
    res = download_dataset("k", tmp_path, ["a/bad", "b/slow", "c/empty", "d/good"], runner=r)
    assert res.slug == "d/good" and not res.skipped and r.calls == list(r.plan)
    assert "SECRET123" not in " ".join(res.tried) and "timeout" in " ".join(res.tried)
    assert (tmp_path / "k" / "data.csv").exists() and not (tmp_path / "k.partial").exists()


def test_download_skips_when_complete_and_redownloads_when_file_missing(tmp_path):
    from chatlens.kaggle_data import download_dataset
    r = FakeRunner({"d/good": "ok"})
    download_dataset("k", tmp_path, ["d/good"], runner=r)
    assert download_dataset("k", tmp_path, ["d/good"], runner=r).skipped and len(r.calls) == 1
    (tmp_path / "k" / "data.csv").unlink()
    assert not download_dataset("k", tmp_path, ["d/good"], runner=r).skipped and len(r.calls) == 2


def test_interrupted_download_never_looks_complete(tmp_path):
    from chatlens.kaggle_data import download_dataset
    part = tmp_path / "k.partial"
    part.mkdir()
    (part / "half.csv").write_text("trunc")               # leftover from a killed runtime
    with pytest.raises(RuntimeError, match="Could not download"):
        download_dataset("k", tmp_path, ["a/bad"], runner=FakeRunner({"a/bad": "fail"}))
    assert not (tmp_path / "k").exists()


def test_all_slugs_failing_message_is_actionable(tmp_path):
    from chatlens.kaggle_data import download_dataset
    with pytest.raises(RuntimeError, match="accepted the dataset terms"):
        download_dataset("k", tmp_path, ["a/bad"], runner=FakeRunner({"a/bad": "fail"}))


def test_auth_detection(monkeypatch, tmp_path):
    from chatlens.kaggle_data import configure_kaggle_auth
    for v in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="No Kaggle credentials"):
        configure_kaggle_auth()
    assert configure_kaggle_auth(token="t") == "access_token"
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    assert configure_kaggle_auth(username="u", key="k") == "legacy_key"


def test_outputs_may_be_declared_after_the_work_and_are_then_checksummed(tmp_path):
    s = CheckpointStore(tmp_path)
    names = []

    def work():
        s.save_json("m/model.json", {"w": 1})
        names.append("m/model.json")
    s.run("fit", work, outputs=lambda: names)
    assert "m/model.json" in s.stages["fit"]["artifacts"]
    (tmp_path / "m/model.json").write_text('{"w": 2}')
    assert "checksum" in CheckpointStore(tmp_path).verify("fit")[1]
