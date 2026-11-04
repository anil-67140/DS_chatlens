# Privacy and security design

| Requirement | Implementation | Test |
|---|---|---|
| Local-first | All processing in-process. The only outbound traffic in the whole code base is **model-weight downloads** (Hugging Face `from_pretrained`) and the **Kaggle CLI**; neither sends chat text. The parse → analytics → dashboard path makes no network calls | verified by source search (`grep` for HTTP/socket/`from_pretrained`); the Docker image runs the dashboard with no outbound access needed |
| No unnecessary raw storage | Uploaded temp file deleted right after it is read, even on failure; working folder 0700; auto-delete after `CHATLENS_RETENTION_MINUTES` | `test_app_service` (deleted, perms, retention) |
| Anonymisation | Names → `Participant N`; names/phones/e-mails in text masked; URL path/query removed (domain kept). Runs **before** feature extraction so tokens/keywords never contain names | `test_validation_privacy`, `test_viz_analyze` |
| Exports | Row-level data has **no message text** unless names are not anonymised *and* the user opts in. Aggregates (n-grams, RAKE phrases) can contain repeated message fragments by design: treat the report as the owner's data | `test_app_service`, `test_report` |
| No content in logs | Loggers receive counts/timings only | `test_logging_never_contains_message_text` |
| Upload validation | Extension, size cap, NUL/binary rejection, encoding fallback, zip: exactly one `.txt`, in-memory only (no extraction → no path traversal), uncompressed-size cap (zip-bomb) | `test_ingestion` |
| Output safety | Everything chat-derived is HTML-escaped in reports and markdown | `test_report`, `test_app_ui` |
| Secrets | Kaggle token via Colab Secrets / env; redacted from error text; never written to Drive; `.env`, `kaggle.json` git-ignored | `test_checkpoint` (redaction) |
| No external NLP | Models run locally. Toxicity refuses to run without `consent=True` | `test_nlp_misc` |
| Container | Non-root, read-only FS, tmpfs scratch, `cap_drop: ALL`, localhost-only port | *Compose file written; image not built in the sandbox* |

**Gaps to know about**: Gradio's own upload cache holds the file briefly before the app deletes its copy (set `delete_cache`); the app has **no authentication**: do not expose it to the internet without a reverse proxy + auth; Colab `share=True` routes traffic through a third-party relay; anonymisation is best-effort (nicknames, addresses, unusual identifiers can survive).
