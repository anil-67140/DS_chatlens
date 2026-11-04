# Dataset research and selection

Goal: larger/better labelled data than the (empty) original project, usable legally, **without replacing the WhatsApp workflow**.

| Dataset | Source | Size | Labels | Language | Licence / terms | Used for | Verdict |
|---|---|---|---|---|---|---|---|
| Roman Urdu sentiment | Kaggle `smat26/sentiment-analysis-on-roman-urdu` (author's repo `Smat26/Roman-Urdu-Dataset`) | 20,229 raw → 19,372 clean | pos/neu/neg, manual | Roman Urdu (≈ Hinglish) | repo is GPL-3.0; check Kaggle page | **Sentiment training/eval for Hinglish chat** | ✅ Core. *Measured in this project* (see `docs/results/`). |
| Emotions dataset for NLP | Kaggle `praveengovi/emotions-dataset-for-nlp` | ~16k/2k/2k | 6 emotions, official splits | English tweets | see Kaggle page | **Emotion** | ✅ Used by the notebook; **not run in the build sandbox** (no Kaggle access). |
| GoEmotions | Kaggle `shivamb/go-emotions-google-emotions-dataset` | ~58k | 27 emotions (unused) | English Reddit | see Kaggle page | Informal-text corpus for **topics/keywords** | ✅ Used by the notebook; not run in sandbox. |
| Sentiment140 | Kaggle `kazanova/sentiment140` | 1.6M | emoticon-derived (**noisy**) | English tweets | see Kaggle page | English sentiment (subsampled to 100k) | ⚠️ Used with the caveat that labels are distant supervision. Not run in sandbox. |
| "WhatsApp chat" datasets | various | — | — | — | personal data of unknown consent / synthetic | — | ❌ **Deliberately not used.** |
| Original project data | `anil-67140/Data-Science-Projects` | 0 rows | — | — | — | — | Empty notebook; nothing to retain. |

Not verified: Hinglish-specific (Devanagari-romanised Hindi) Kaggle corpora and Urdu-script sentiment corpora were searched for but no slug was verified, so none is used. Slugs above are verified listings; Kaggle can rename/remove datasets, which the downloader reports clearly (and Roman Urdu falls back to its GitHub mirror).

**None of these is representative of real WhatsApp conversations** (banter, sarcasm, code-switching, one-word replies). The sentiment numbers are therefore *indicative*; to know in-domain accuracy you must label a sample of your own chat messages.
