# Sentiment on Roman Urdu — measured results (sandbox run)

**Provenance.** Real data, real run, executed in the build sandbox (CPU, no GPU, no Hugging Face access) with
chatlens 0.2.0, seed 42. Dataset: *Roman Urdu* sentiment corpus (Positive / Neutral / Negative), mirrored at
`Smat26/Roman-Urdu-Dataset` (GPL-3.0) and on Kaggle as `smat26/sentiment-analysis-on-roman-urdu`.
**Not included here:** the pretrained and fine-tuned transformer methods. They need a GPU and model downloads and
are run by the Colab notebook; this file will not contain numbers for them until you run it.

## Data cleaning (every dropped row accounted for)
| Step | Rows |
|---|---|
| Raw rows read | 20,229 |
| Missing/blank text | −140 |
| Label typo repaired (`Neative`→`negative`) | 1 repaired (kept) |
| Same text with conflicting labels (all copies removed) | −154 |
| Duplicate texts (normalised; first kept) | −520 |
| **Clean rows** | **19,372** |

Class counts: {'negative': 5231, 'neutral': 8347, 'positive': 5794}. Split (stratified, seed 42, leakage-checked): train 13,559 / val 1,938 / test 3,875.
De-duplicating *before* splitting matters: otherwise the same sentence can sit in train and test and inflate scores.

## Method comparison (test set, n = 3,875)
Hyper-parameters were selected on the validation split only. CI = percentile bootstrap (500 resamples of the test items).
Lexicons produce no probabilities, so AUC/ECE are not defined for them (shown as —).

| # | Method | Family | Accuracy | Macro-F1 [95% CI] | ROC-AUC (OvR) | ECE | Fit (s) |
|---|---|---|---|---|---|---|---|
| 1 | `tfidf_word+char_logreg` | TF-IDF | 0.670 | **0.664** [0.650, 0.678] | 0.830 | 0.015 | 21 |
| 2 | `tfidf_char_logreg` | TF-IDF | 0.657 | **0.650** [0.635, 0.664] | 0.823 | 0.016 | 20 |
| 3 | `tfidf_word+char_linearsvc_calibrated` | TF-IDF | 0.663 | **0.649** [0.634, 0.663] | 0.832 | 0.050 | 12 |
| 4 | `tfidf_word+char_sgd_modhuber` | TF-IDF | 0.654 | **0.646** [0.631, 0.661] | 0.812 | 0.057 | 9 |
| 5 | `tfidf_word_logreg` | TF-IDF | 0.643 | **0.639** [0.624, 0.652] | 0.810 | 0.031 | 5 |
| 6 | `tfidf_word_complementnb` | TF-IDF | 0.618 | **0.615** [0.599, 0.631] | 0.795 | 0.140 | 3 |
| 7 | `tfidf_word_multinomialnb` | TF-IDF | 0.615 | **0.606** [0.591, 0.622] | 0.784 | 0.056 | 3 |
| 8 | `tfidf_word_randomforest` | TF-IDF | 0.617 | **0.591** [0.574, 0.608] | 0.779 | 0.035 | 87 |
| 9 | `vader_tuned` | lexicon, tuned band | 0.388 | **0.322** [0.306, 0.335] | — | — | 1 |
| 10 | `vader` | zero-shot lexicon | 0.388 | **0.321** [0.306, 0.334] | — | — | 0 |
| 11 | `textblob_tuned` | lexicon, tuned band | 0.417 | **0.307** [0.292, 0.320] | — | — | 2 |
| 12 | `textblob` | zero-shot lexicon | 0.417 | **0.305** [0.290, 0.319] | — | — | 0 |
| 13 | `majority_baseline` | baseline | 0.431 | **0.201** [0.196, 0.206] | 0.500 | 0.000 | 0 |

## Is the winner really better? (paired, vs `tfidf_word+char_logreg`)
| Method | Δ macro-F1 vs best | Holm-adj. p | Verdict |
|---|---|---|---|
| `majority_baseline` | +0.463 [+0.448, +0.479] | 0.0000 | reference better |
| `vader` | +0.343 [+0.322, +0.364] | 0.0000 | reference better |
| `vader_tuned` | +0.342 [+0.322, +0.363] | 0.0000 | reference better |
| `textblob` | +0.359 [+0.337, +0.378] | 0.0000 | reference better |
| `textblob_tuned` | +0.357 [+0.335, +0.376] | 0.0000 | reference better |
| `tfidf_word_logreg` | +0.025 [+0.015, +0.038] | 0.0000 | reference better |
| `tfidf_char_logreg` | +0.014 [+0.003, +0.024] | 0.0181 | reference better |
| `tfidf_word+char_linearsvc_calibrated` | +0.015 [+0.007, +0.025] | 0.1391 | indistinguishable |
| `tfidf_word_complementnb` | +0.049 [+0.033, +0.064] | 0.0000 | reference better |
| `tfidf_word_multinomialnb` | +0.058 [+0.042, +0.073] | 0.0000 | reference better |
| `tfidf_word+char_sgd_modhuber` | +0.018 [+0.010, +0.028] | 0.0008 | reference better |
| `tfidf_word_randomforest` | +0.073 [+0.057, +0.089] | 0.0000 | reference better |

## What this does and does not show
* The best model reaches **macro-F1 0.664** (accuracy 0.670). Per-class F1: negative 0.63, neutral 0.70, positive 0.65. Roman Urdu sentiment is a hard, noisy task (the neutral class is the largest and the labels are subjective); this is a fair ceiling for bag-of-n-gram models, not a deficiency to hide.
* **English lexicons (VADER, TextBlob) are barely above the majority baseline** on macro-F1 and *below* it on accuracy: they do not understand Roman Urdu. This is why the app never applies them to Hinglish/Urdu text.
* The best three supervised methods are within a few points; `tfidf_word+char_linearsvc_calibrated` is **statistically indistinguishable** from the best, so "best" is not a strong claim. Choosing LR is justified by better calibration (ECE 0.015) and simplicity, not by a clear accuracy win.
* The model is trained on e-commerce/social-media style Roman Urdu. WhatsApp banter, sarcasm and code-mixing are outside that distribution; expect lower accuracy in the wild. No WhatsApp-specific labelled sentiment corpus was used, so **in-domain accuracy is unknown**.
* Confusion matrix (rows = truth): [[660, 263, 123], [220, 1187, 263], [153, 256, 750]] with label order ['negative', 'neutral', 'positive'].
