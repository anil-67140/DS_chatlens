import pandas as pd
import pytest

from chatlens.kaggle_data import (
    assert_no_leakage,
    clean_labeled,
    load_emotions,
    load_roman_urdu,
    load_sentiment140,
    split_dataset,
    text_key,
)


def test_text_key_normalisation():
    assert text_key("Wah!! kya   BAAT hai.") == text_key("wah kya baat hai")
    assert text_key("see https://x.com/a now") == text_key("see now")


def test_clean_accounts_for_every_row():
    raw = pd.DataFrame({
        "text": ["good", "Good!", "bad", None, " ", "same", "same", "ok fine"],
        "label": ["Positive", "Positive", "Neative", "Positive", "Positive",
                  "Positive", "Negative", "Neutral"]})
    d, r = clean_labeled(raw)
    assert r.n_raw == 8 and r.dropped_missing_text == 2 and r.labels_repaired == 1
    assert r.dropped_conflicting == 2          # 'same' with two labels is removed entirely
    assert r.dropped_exact_dup == 1            # 'Good!' duplicates 'good'
    assert (r.n_raw - r.dropped_missing_text - r.dropped_missing_label
            - r.dropped_conflicting - r.dropped_exact_dup) == r.n_clean == len(d)
    assert set(d["label"]) == {"positive", "negative", "neutral"}     # 'Neative' repaired


def test_split_is_stratified_leak_free_and_reproducible():
    n = 600
    df = pd.DataFrame({"text": [f"sentence number {i} about topic {i % 7}" for i in range(n)],
                       "label": ["a", "b", "c"] * (n // 3)})
    s1, s2 = split_dataset(df, seed=1), split_dataset(df, seed=1)
    assert sum(len(v) for v in s1.values()) == n
    pd.testing.assert_frame_equal(s1["test"], s2["test"])
    for part in s1.values():
        assert part["label"].value_counts(normalize=True).max() == pytest.approx(1 / 3, abs=0.03)
    assert abs(len(s1["test"]) / n - 0.2) < 0.01 and abs(len(s1["val"]) / n - 0.1) < 0.01


def test_leakage_detected():
    a = pd.DataFrame({"text": ["Hello there!"], "label": ["x"]})
    b = pd.DataFrame({"text": ["hello there"], "label": ["x"]})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage({"train": a, "test": b})


def test_loaders(tmp_path):
    (tmp_path / "Roman Urdu DataSet.csv").write_text(
        "wah kya baat hai,Positive,\nbura sauda,Negative,\nthik hai,Neutral,\n", encoding="utf-8")
    ru = load_roman_urdu(tmp_path)
    assert list(ru.columns) == ["text", "label", "source"] and len(ru) == 3
    for n, body in (("train.txt", "i am happy;joy\nso sad;sadness\n"), ("val.txt", "wow;surprise\n"),
                    ("test.txt", "grr;anger\n")):
        (tmp_path / n).write_text(body)
    em = load_emotions(tmp_path)
    assert {k: len(v) for k, v in em.items()} == {"train": 2, "val": 1, "test": 1}
    (tmp_path / "training.1600000.processed.noemoticon.csv").write_bytes(
        b'0,1,d,NO_QUERY,u,"sad caf\xe9 day"\n4,2,d,NO_QUERY,u,"great day"\n')
    s140 = load_sentiment140(tmp_path, max_rows=None)
    assert set(s140["label"]) == {"negative", "positive"}


def test_real_roman_urdu_if_available():
    """Runs on the real 20k-row corpus when it was cloned to /tmp/rud (local dev only)."""
    import os
    p = "/tmp/rud/Dataset"
    if not os.path.isdir(p):
        pytest.skip("real dataset not present")
    d, r = clean_labeled(load_roman_urdu(p))
    assert r.n_raw > 20000 and r.labels_repaired >= 1 and r.dropped_exact_dup > 100
    splits = split_dataset(d)
    assert_no_leakage(splits)
