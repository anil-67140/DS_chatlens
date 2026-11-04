# Methodology, assumptions and limitations (per feature)

| Feature | Method | Key assumptions / limitations |
|---|---|---|
| **Parsing** | Regex headers for Android (12h/24h) and iOS `[..]` exports; continuation lines are appended to the previous message | Day/month order is *inferred* (any value >12 is decisive; otherwise fewer chronological regressions wins; ties → day-first and a warning). Only English system strings/media markers are built in; other locales need `extra_media_markers`. **Never verified against a real export from your phone**—run it and read the parse report. |
| **Accounting** | Every header line → exactly one row | Invalid timestamps are kept (flagged), duplicates flagged not removed (legitimate repeats exist). |
| **Language ID** | Script (Arabic→`ur`, Devanagari→`hi`) + Roman-Hindi/Urdu word list for Hinglish vs English | Heuristic, unbenchmarked. Arabic script may be Arabic/Persian; other Latin-script languages are labelled `en`. |
| **Stopwords** | Small built-in lists (en, Hinglish, hi, ur) | Not exhaustive; Hinglish has no standard spelling. |
| **Sessions** | New session after ≥ `session_gap_minutes` (60) of silence | Threshold is a modelling choice; configurable. |
| **Response times** | First message of a new speaker after another speaker, measured from the other speaker's *last* message; counted only if ≤ `response_window_minutes` (360) | **Inference from adjacency**: exports have no reply links or read receipts. In groups the reply may be to someone else. 1-minute timestamp resolution. Median CI: bootstrap (n ≤ 5000) or exact order-statistic interval. |
| **"No observed reply"** | A turn not followed by another speaker within the window; final turn excluded | Does **not** mean ignored (read, phoned, replied elsewhere). |
| **Concentration** | Gini, HHI, normalised entropy, top-1/top-3 share | In 2-person chats these only reflect who talks slightly more. |
| **Trend / seasonality** | Theil–Sen slope + Spearman on weekly volume; month-of-year profile | Trend needs ≥ 8 weeks; seasonality labelled reliable only with ≥ 24 months. |
| **Group tests** | Kruskal–Wallis (ε²), Mann–Whitney (rank-biserial), χ² (Cramér's V), Spearman | Messages are autocorrelated, so p-values are optimistic: use effect sizes. Non-parametric because lengths/response times are heavy-tailed. |
| **Sentiment (chat)** | Messages routed by detected language to a registered model; others `unscored` | Default: VADER for English (zero-shot), TF-IDF+LR (trained on Roman Urdu corpus) for Hinglish. Urdu/Hindi *script* is unscored unless you register a multilingual model. Accuracy = the benchmark's, not 100%; WhatsApp-domain accuracy is unknown. |
| **Topics (chat)** | NMF on *conversation sessions* (one document per session); k from coherence within tolerance, ties by silhouette | Needs ≥ 60 usable sessions. Labels are top words only. Coherence is a proxy for interpretability. |
| **Keywords** | TF-IDF, RAKE, YAKE, TextRank, embedding-MMR, frequency | No gold standard → compared by stability and agreement, not accuracy. Short repeated phrases can be whole messages. |
| **Summaries** | Extractive: TF-IDF centroid + MMR | Selects real messages; cannot paraphrase. |
| **NER / toxicity** | Optional library functions (spaCy / HF), toxicity consent-gated | Not wired into the dashboard; NER is weak on informal/code-mixed text; toxicity models are biased. Untested against real models here. |

## Evaluation metrics: which apply where
* **Classification (sentiment, emotion)**: accuracy, balanced accuracy, per-class P/R/F1, macro/weighted F1 (+bootstrap CI), MCC, Cohen's κ, confusion matrix; **ROC-AUC, PR-AUC, ECE, Brier, log-loss only when the method outputs probabilities** (lexicons do not; those cells are empty, never invented).
* **Model comparison**: paired McNemar (exact for few discordant pairs), paired bootstrap CI of the macro-F1 difference, Holm correction.
* **Topic / clustering (unsupervised)**: NPMI coherence, topic diversity, silhouette, Davies–Bouldin, Calinski–Harabasz. No accuracy, ever.
* **Keywords**: bootstrap stability, Jaccard agreement, runtime.
