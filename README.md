# chatlens — WhatsApp Chat Intelligence (work in progress)

Privacy-first parsing, validation and analytics for exported WhatsApp chats.
Individual chats and groups; Android and iOS exports; English, Urdu, Hindi and Hinglish text.

> **Status: Phases 1–7 of 12 complete (foundation + analytics core).**
> Not yet built: NLP model comparison, Colab notebook, Gradio app, Docker, CI, Kaggle dataset work.
> See the table below. Nothing in this README describes a feature that does not exist in the code.

## What is implemented and tested

| Area | Module | Notes |
|---|---|---|
| Safe ingestion | `ingestion.py` | `.txt` / single-`.txt` `.zip`; size cap, zip-bomb guard, in-memory only, NUL rejection, encoding fallback |
| Parsing | `parsing/` | Android 12h/24h, iOS `[..]`, `/ . -` separators, ISO dates, multi-line, system/media/deleted/edited/location, day-vs-month inference |
| Accounting | `ParseReport` | every header → exactly one row; invalid timestamps kept + flagged; duplicates flagged, never removed |
| Preprocessing | `preprocessing/` | URLs, domains, emoji (ZWJ/skin-tone aware), Unicode normalisation, Urdu/Hindi-safe tokenisation, per-message language, language-specific stopwords |
| Validation | `validation.py` | plausibility, chronology, sender sanity, parse coverage |
| Analytics | `analytics/` | overview, temporal (heatmap, streaks, silences, trend, seasonality), participants (Gini/HHI/entropy), behaviour (sessions, initiation, response times, no-observed-reply, interaction matrix), text (words, n-grams, emoji, domains, distinctive vocabulary) |
| Statistics | `stats.py` | Kruskal–Wallis (ε²), Mann–Whitney (rank-biserial), χ² (Cramér's V), Spearman, Theil–Sen, bootstrap / order-statistic CIs; each with a magnitude label |
| Privacy | `privacy.py`, logging | anonymisation + text scrubbing; logs contain stage timings and counts, never message text (tested) |

## Quick start

```bash
pip install -e ".[dev]"
pytest --cov=chatlens
```
```python
from chatlens import read_chat, parse_chat, preprocess, validate
from chatlens.analytics import overview, temporal, behavior

ing = read_chat("WhatsApp Chat.txt")
res = parse_chat(ing.text)          # res.report tells you what was parsed / flagged and why
df = preprocess(res.df)
print(validate(df, res.report).to_dict())
print(overview.overview(df)["total_messages"], behavior.response_summary(df)["overall"])
```

## Verification

* 87 tests, 96% line coverage, `ruff` clean.
* Analytics are tested against a **synthetic** chat whose correct answers are known by construction
  (e.g. every reply is exactly 5 or 3 minutes, so the median must be 5 / 3).
* Scale: 300 000 messages (16.7 MB) parse + preprocess + all analytics ≈ 35 s, ≈ 770 MB peak RSS
  on the build sandbox. (Benchmark found and fixed an O(iterations × n) memory blow-up in the bootstrap.)

## Methodology and limitations (current features)

* **Response times** are inferred from message adjacency. Exports have no reply links or read
  receipts; in groups the reply may be addressed to someone else. Replies later than
  `response_window_minutes` (default 6 h) are excluded and counted separately. "No observed reply"
  ≠ ignored. Timestamps have 1-minute resolution.
* **Language ID** is a script + Roman-Hindi/Urdu lexicon heuristic: Arabic script → `ur` (may be
  Arabic/Persian), Devanagari → `hi` (may be Marathi/Nepali), other Latin-script languages → `en`.
  Not yet benchmarked against a trained model.
* **Stopword lists are small and conservative**; Hinglish has no standard spelling.
* **Localised exports**: English system strings and media markers are built in. Other locales need
  `extra_media_markers` and may mislabel system lines as messages. This is untested on real
  non-English-locale exports.
* **No real WhatsApp export has been used in testing** (privacy; none was provided). Formats are
  based on documented layouts, not verified against your device. Please run it on one of your
  own chats and report parse warnings.
* **Anonymisation is best-effort**: free text can contain identifying details that are not names,
  numbers, e-mails or URLs.
* Statistical tests assume independent observations; chat messages are autocorrelated, so p-values are
  optimistic. Effect sizes are reported for that reason.

## Roadmap (not implemented)

| # | Phase | Status |
|---|---|---|
| 1 | Audit & gap analysis | ✅ `docs/00_audit_and_gap_analysis.md` |
| 2 | Dataset research (needs Kaggle/network access) | ⏳ |
| 3 | Architecture | ✅ package layout + decisions in docs |
| 4–5 | Ingestion, preprocessing, EDA | ✅ |
| 6 | NLP: sentiment / emotion / topics / keywords / NER, ≥10-method comparison with labelled data | ⏳ |
| 7 | Statistical analysis | ✅ (descriptive + inferential helpers); report generation ⏳ |
| 8 | Colab notebook, Drive, checkpoints, manifest | ⏳ |
| 9 | Gradio frontend | ⏳ |
| 10 | Further QA | ongoing |
| 11 | Docker, Compose, GitHub Actions | ⏳ |
| 12 | Final docs, screenshots | ⏳ |
