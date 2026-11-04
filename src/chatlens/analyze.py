"""One-call analysis of a WhatsApp export. Used by the Gradio app, the notebook and the tests so
there is a single code path to verify."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import __version__, stats
from .analytics import behavior, overview, participants, temporal
from .analytics import text as tx
from .config import AnalysisConfig
from .monitoring import nlp_output_checks, resource_monitor
from .nlp import keywords as kw
from .nlp import summarize, topics
from .nlp.chat_nlp import LanguageRouter, summarise_sentiment
from .parsing.parser import parse_chat
from .preprocessing.pipeline import preprocess
from .privacy import anonymize
from .validation import infer_chat_type, validate


@dataclass
class NLPOptions:
    sentiment: LanguageRouter | None = None
    emotion: LanguageRouter | None = None
    run_topics: bool = True
    run_keywords: bool = True
    run_summaries: bool = True
    topic_ks: tuple[int, ...] = (3, 5, 8)
    min_docs_for_topics: int = 60
    reference_sentiment_dist: dict[str, float] | None = None
    keyword_methods: tuple[str, ...] = ("frequency", "tfidf", "rake", "yake", "textrank")
    encoder: object | None = None


@dataclass
class AnalysisBundle:
    meta: dict
    parse_report: dict
    quality: dict
    df: pd.DataFrame
    overview: dict
    participants: pd.DataFrame
    concentration: dict
    comparison_tests: dict
    temporal: dict
    behavior: dict
    text: dict
    nlp: dict
    timings: dict
    private: dict = field(default_factory=dict)      # never exported (e.g. name mapping)


def session_documents(df: pd.DataFrame, config: AnalysisConfig, min_messages: int = 2) -> tuple[list[str], list[int]]:
    """One document per conversation session (concatenated text messages). Chat messages are far too
    short for topic models; sessions are the natural unit of 'a conversation about something'."""
    um = behavior.add_sessions(df, config)
    um = um[um["is_text"]]
    g = um.groupby("session_id")
    docs, ids = [], []
    for sid, grp in g:
        if len(grp) >= min_messages:
            docs.append(" . ".join(grp.sort_values("timestamp")["message"].tolist()))
            ids.append(int(sid))
    return docs, ids


def analyze_chat(raw_text: str, config: AnalysisConfig | None = None, anonymize_names: bool = False,
                 nlp: NLPOptions | None = None, source_name: str = "upload") -> AnalysisBundle:
    cfg = config or AnalysisConfig()
    opts = nlp or NLPOptions()
    timings: dict = {}
    with resource_monitor(timings, "parse"):
        res = parse_chat(raw_text, cfg)
    private: dict = {}
    parsed = res.df
    if anonymize_names:                     # BEFORE preprocessing: derived features come from scrubbed text
        parsed, mapping = anonymize(parsed, mask_text=True)
        private["name_mapping"] = mapping
    with resource_monitor(timings, "preprocess"):
        df = preprocess(parsed, cfg)
    with resource_monitor(timings, "validate"):
        quality = validate(df, res.report)
    chat_type = infer_chat_type(df)

    with resource_monitor(timings, "analytics"):
        ov = overview.overview(df, include_text=False)
        ptab = participants.participant_table(df)
        conc = participants.concentration(df)
        comp = participants.compare_participants(df, cfg.alpha)
        tmp = {"hourly": temporal.hourly(df), "weekday": temporal.weekday(df), "monthly": temporal.monthly(df),
               "weekly": temporal.weekly(df), "heatmap": temporal.heatmap(df), "daily": temporal.daily_counts(df),
               "weekend_vs_weekday": temporal.weekend_vs_weekday(df, cfg.alpha), "top_hours": temporal.top_hours(df),
               "gaps": temporal.longest_inactive_periods(df), "streaks": temporal.streaks(df),
               "trend": temporal.trend(df), "seasonality": temporal.seasonality(df)}
        beh = {"sessions": behavior.sessions_summary(df, cfg), "initiation": behavior.initiation(df, cfg),
               "response_summary": behavior.response_summary(df, cfg),
               "response_distribution": behavior.response_distribution(df, cfg),
               "no_observed_reply": behavior.no_observed_reply(df, cfg),
               "interaction_matrix": behavior.interaction_matrix(df, cfg),
               "exchange": behavior.exchange_stats(df, cfg)}
        txt = {"top_words": tx.top_words(df, cfg.top_n),
               "bigrams": tx.ngrams(df, 2, cfg.min_ngram_count, cfg.top_n),
               "trigrams": tx.ngrams(df, 3, cfg.min_ngram_count, cfg.top_n),
               "emoji": tx.emoji_frequency(df, cfg.top_n), "emoji_by_sender": tx.emoji_by_sender(df),
               "domains": tx.domain_frequency(df, cfg.top_n), "length": tx.length_distribution(df),
               "languages": tx.language_distribution(df), "distinctive": tx.distinctive_vocabulary(df),
               "frequent_messages": tx.frequent_messages(df, cfg.min_ngram_count, cfg.top_n)}

    nlp_out: dict = {"notes": []}
    n_text = int((df["is_text"] & ~df["is_system"]).sum())
    if opts.sentiment is not None:
        with resource_monitor(timings, "sentiment"):
            scored = opts.sentiment.score(df)
            summ = summarise_sentiment(df, scored, cfg.alpha)
            sc = scored[scored["label"] != "unscored"]
            summ["monitoring"] = nlp_output_checks(
                sc["label"].tolist(), sc["confidence"].dropna().to_numpy() if sc["confidence"].notna().any() else None,
                opts.reference_sentiment_dist or {"positive": 1 / 3, "neutral": 1 / 3, "negative": 1 / 3},
                n_total=n_text)
            nlp_out["sentiment"] = summ
            nlp_out["sentiment_scores"] = scored
    if opts.emotion is not None:
        with resource_monitor(timings, "emotion"):
            scored = opts.emotion.score(df)
            nlp_out["emotion"] = {"scores": scored,
                                  "counts": scored[scored["label"] != "unscored"]["label"].value_counts().to_dict(),
                                  "coverage": float((scored["label"] != "unscored").mean()) if len(scored) else 0.0}
    if opts.run_topics or opts.run_summaries or opts.run_keywords:
        docs, sids = session_documents(df, cfg)
    if opts.run_topics:
        with resource_monitor(timings, "topics"):
            clean, toks, keep = topics.prepare_docs(docs)
            if len(clean) < opts.min_docs_for_topics:
                nlp_out["topics"] = {"skipped": f"only {len(clean)} usable sessions (< {opts.min_docs_for_topics}); "
                                                "topic models need more text"}
            else:
                ks = tuple(k for k in opts.topic_ks if k < len(clean) // 5) or (2,)
                space = topics.evaluation_space(clean, cfg.seed)
                rows = topics.sweep_k("nmf", clean, toks, ks, cfg.seed, space)
                best = topics.best_k(rows)
                r = best["result"]
                nlp_out["topics"] = {
                    "method": "nmf", "k": best["k"],
                    "sweep": [{k: v for k, v in x.items() if k not in ("result", "per_topic_npmi")} for x in rows],
                    "topics": r.topics, "sizes": np.bincount(r.labels[r.labels >= 0], minlength=len(r.topics)).tolist(),
                    "npmi_coherence": best["npmi_coherence"], "topic_diversity": best["topic_diversity"],
                    "n_sessions_used": len(clean), "unit": "conversation session",
                    "caveat": "Topic labels are the top words only; read them before trusting them."}
    if opts.run_keywords:
        with resource_monitor(timings, "keywords"):
            tm = df[df["is_text"] & ~df["is_system"]]["message"].tolist()
            if len(tm) >= 30:
                kr = kw.run_all(tm[:20000], cfg.top_n, list(opts.keyword_methods), opts.encoder, stability_runs=2,
                                seed=cfg.seed)
                nlp_out["keywords"] = kr
            else:
                nlp_out["keywords"] = {"skipped": "fewer than 30 text messages"}
    if opts.run_summaries:
        with resource_monitor(timings, "summaries"):
            nlp_out["summaries"] = summarize.summarize_chat(df, cfg, top_sessions=5, k=3, include_text=False)

    meta = {"chatlens_version": __version__, "source": source_name, "chat_type": chat_type,
            "anonymized": anonymize_names, "n_rows": len(df), "config": cfg.to_dict(),
            "date_order": res.report.date_order, "format": res.report.source_format}
    return AnalysisBundle(meta, res.report.to_dict(), quality.to_dict(), df, ov, ptab, conc, comp, tmp, beh, txt,
                          nlp_out, timings, private)


def message_level_tests(bundle: AnalysisBundle) -> dict:
    """Convenience: the inferential results gathered in one place for the stats report."""
    out = {"participant_comparison": bundle.comparison_tests,
           "weekend_vs_weekday": bundle.temporal.get("weekend_vs_weekday"),
           "trend": bundle.temporal.get("trend"),
           "response_time_difference": bundle.behavior["response_summary"].get("difference_between_responders")}
    s = bundle.nlp.get("sentiment") or {}
    out["sentiment_participant_difference"] = s.get("participant_difference_test")
    return out


__all__ = ["AnalysisBundle", "NLPOptions", "analyze_chat", "message_level_tests", "session_documents", "stats"]
