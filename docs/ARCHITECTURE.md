# Architecture

```
upload ─► ingestion ─► parsing ─► [anonymize] ─► preprocessing ─► validation
                                                      │
        ┌─────────────────────────────────────────────┼──────────────────────────┐
        ▼                         ▼                    ▼                          ▼
  analytics/ (overview,      nlp/ (language-routed   viz.py (spec → Plotly +   report.py
  temporal, participants,    sentiment, topics on    Matplotlib)               (escaped HTML,
  behaviour, text) + stats   sessions, keywords,                                exports)
                             summaries, similarity)
        └───────────────► analyze.py (AnalysisBundle) ◄────── app/service.py ◄── app/gradio_app.py (+ /healthz)

Offline experiments (Colab):  kaggle_data ─► experiments.prepare_dataset ─► nlp.benchmark (per-method checkpoints)
                              ─► topic/keyword benchmarks ─► export_model + tracking (MLflow) ─► report.project_report
Cross-cutting: config.py · checkpoint.py (atomic writes, manifest, SHA-256) · logging_utils.py · monitoring.py · privacy.py
```

| Layer | Modules |
|---|---|
| Data ingestion / validation | `ingestion`, `parsing/`, `validation`, `kaggle_data` |
| Preprocessing / feature engineering | `preprocessing/` (normalise, tokenise, emoji, URLs, language, stopwords) |
| Exploratory analysis | `analytics/` |
| NLP | `nlp/classifiers`, `nlp/benchmark`, `nlp/evaluation`, `nlp/topics`, `nlp/keywords`, `nlp/summarize`, `nlp/similarity`, `nlp/ner`, `nlp/toxicity`, `nlp/chat_nlp` |
| Statistics | `stats` |
| Visualisation / reporting | `viz`, `report` |
| Application | `app/service` (logic), `app/render` (display values), `app/gradio_app` (UI + health) |
| Ops | `checkpoint`, `tracking`, `monitoring`, `config`, `logging_utils` |
| Quality | `tests/` (226), `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml` |

## Decisions
* **Gradio, not Flask.** The product is upload → tabs of tables/plots → download. Gradio supplies uploads, progress, queueing and Plotly components, and runs in Colab and Spaces unchanged; Flask would mean hand-writing routing, async job handling and a JS chart layer for no analytical gain. The app is mounted in FastAPI to add `/healthz`. Choose Flask only if you need custom auth or an existing backend.
* **One analysis code path** (`analyze_chat`) shared by app, notebook and tests.
* **Per-method checkpoint stages** so a Colab disconnect costs at most one method (fine-tuning resumes per epoch).
* **Code version in the checkpoint key** so stale results from older code are never reused.
* **No DVC / Sentry**: see `MONITORING_AND_MODEL_UPDATES.md`.
* **Plain PyTorch fine-tuning loop** instead of `Trainer`, to avoid API drift and to control Drive checkpointing.
