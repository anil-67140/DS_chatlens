# Phase 1 — Repository Audit & Gap Analysis

**Repository:** `anil-67140/Data-Science-Projects` → `campus-x-projects/05-wattsapp-chat-analysis`
**Method:** full `git clone` (the GitHub web page blocks automated fetching), full history (`git log`),
recursive file listing, parsing of the notebook JSON.

## What exists

| Item | Finding |
|---|---|
| Files in the project folder | **1** — `05-wattsappchat_alalysis.ipynb` (324 bytes) |
| Notebook cells | **1 code cell, empty** (no source, no outputs) |
| Dataset / sample chat | none (no `.txt`, no `.csv` anywhere in the repo mentions WhatsApp) |
| Python modules, requirements, README, tests | none |
| Git history | one commit, `initial upload` — no earlier version holds content |
| Sibling projects (02, 03, 04, 06) | Notebook + CSV + pickled model each; no packaging, tests or docs either |

## Conclusion

**There is no existing implementation to audit, keep or fix.** The folder is a blank Colab
notebook (the Colab metadata shows it was created but never filled in, or the work was never pushed).
This is therefore a **greenfield build**, not an upgrade. Nothing was rewritten or deleted because
there was nothing to rewrite. If you have the real notebook locally (e.g. the CampusX tutorial
version), please upload it: I would then port any logic you want kept and diff it against this design.

## Gap analysis (everything is a gap, so prioritised by dependency order)

| Priority | Gap | Status in this delivery |
|---|---|---|
| **Critical** | Parser for WhatsApp exports (formats, multi-line, system/media, date order) | ✅ built + tested |
| **Critical** | Safe ingestion (size/zip-bomb/binary/encoding) | ✅ built + tested |
| **Critical** | Never silently drop data; parse accounting | ✅ `ParseReport`, invariant test |
| **Critical** | Privacy: no content in logs, anonymisation | ✅ built + tested (see limits) |
| **High** | Preprocessing (URLs, emoji, normalisation, language, stopwords) | ✅ built + tested |
| **High** | Data-quality validation | ✅ built + tested |
| **High** | EDA: overview, temporal, participants, behaviour, text | ✅ built + tested against ground truth |
| **High** | Statistics with effect sizes | ✅ built + tested |
| **High** | Modular package, config, type hints, lint, tests | ✅ |
| **High** | NLP: sentiment/emotion/topic/keyword/NER + ≥10-method comparison | ⏳ next phase |
| **High** | Colab notebook + Drive + checkpointing + manifest | ⏳ next phase |
| **High** | Gradio frontend | ⏳ next phase |
| **Medium** | MLflow tracking, DVC | ⏳ (only for the NLP experiments, where it earns its keep) |
| **Medium** | Docker, Compose, GitHub Actions CI | ⏳ |
| **Medium** | Dataset research / Kaggle download | ⏳ needs network access this sandbox lacks |
| **Optional** | Drift monitoring, feedback collection | ⏳ only for trained-model tasks |

## Decisions recorded

* **Frontend: Gradio**, not Flask. Reason: the product is a data-exploration dashboard
  (upload → tabs of plots/tables → download), which Gradio provides with built-in file upload,
  progress, queueing and Plotly support, and it runs in Colab and Hugging Face Spaces unchanged.
  Flask would mean hand-writing routing, templates, async job handling and a JS charting layer
  for no analytical gain. Flask becomes the better choice only if you need custom auth or an
  existing web backend.
* **Behaviour metrics are inference, not fact** — see the assumptions block in
  `analytics/behavior.py`; "no observed reply" is never described as "ignored".
* **Language ID is a transparent script + lexicon heuristic**, documented as such, and to be
  benchmarked against alternatives in the NLP phase rather than assumed accurate.


---
## Update (phase 2): your new repository `anil-67140/DS_chatlens`
Cloned and inspected: it contained exactly the v0.1 package from the first delivery (one commit). Nothing else to preserve.
All gap-analysis items above marked ⏳ are now implemented; see the README status table for what is verified vs only written.
Gaps found *during* the build and fixed (each has a regression test): UTF-16 mis-decoding; O(iterations×n) bootstrap memory blow-up;
unicode-unaware vectorisers splitting Hindi/Urdu words; empty-mask pandas bug crashing degenerate chats; anonymisation blanking
text features; model files missing from checksum verification; calibration failure on scikit-learn 1.4.
