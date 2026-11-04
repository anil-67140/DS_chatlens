# Monitoring, feedback and model-update process

| Concern | What exists | Notes |
|---|---|---|
| Processing time & resources | `resource_monitor` records wall/CPU/RSS per stage; shown in the report | |
| Data-quality monitoring | `validate()` (plausibility, chronology, sender sanity, parse coverage) + `ParseReport` | Applies to every upload. |
| NLP output quality | `nlp_output_checks`: coverage, single-class dominance, confidence, PSI vs reference | Alerts appear in the sentiment tab/report. |
| Drift monitoring | PSI of predicted-label distribution vs the model's held-out reference (`models/reference_distribution.json`) | **Only for the trained sentiment model.** No drift monitoring is attempted for the parser or frequency analytics (no model to drift). Drift ⇒ "this chat is unlike the evaluation data": a reason to distrust output, not proof of error. |
| Model/dataset versioning | `dataset_fingerprint` (content hash), package version in every checkpoint key, model card JSON, MLflow runs (SQLite, synced to Drive) | DVC was **not** added: the manifest + fingerprints + MLflow already give versioned, hash-verified data and runs in Colab; DVC would add a remote and workflow without new information. |
| Error monitoring | Structured logs; user-facing errors are typed; no Sentry/third-party reporting (would transmit data) | Add one only if you accept that trade-off. |
| Feedback | Dashboard tab → `FeedbackStore` JSONL (section, 1–5 rating, ≤500-char comment; paste-detection), mode 0600 | Never receives chat content. |

## Updating the model (documented procedure)
1. Collect new labelled data (ideally a sample of *your own* messages, hand-labelled). 2. Re-run notebook §5–§7 (checkpoints make unchanged parts free).
3. Candidate must beat the incumbent on the **significance table** (Holm-adjusted, CI excludes 0) *and* not worsen ECE. 4. Export (§12) – this refuses unverified checkpoints.
5. Bump `__version__` (invalidates cached stages), commit `models/*.joblib` + card, tag a release. 6. Re-run drift check on a recent chat before announcing.
