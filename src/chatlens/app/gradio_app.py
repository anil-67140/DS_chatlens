"""Gradio dashboard + FastAPI wrapper (health check). Run: ``python -m chatlens.app.gradio_app``."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from ..monitoring import FeedbackStore
from ..synthetic import make_synthetic_chat
from . import render as R
from .service import AppSettings, UserFacingError, process_upload

PRIVACY = ("🔒 **Privacy**: your file is processed in memory on the machine running this app and the uploaded copy is "
           "deleted immediately. Nothing is sent to any external service. Names are anonymised by default. Results "
           "are kept for a short time so you can download them, then deleted.")
SECTIONS = ["overview", "activity", "participants", "behaviour", "text", "sentiment", "topics", "statistics", "other"]
FIG_KEYS = ["activity_heatmap", "hourly", "weekday", "timeline", "participants", "participant_trend", "response_times",
            "interaction", "top_words", "emoji", "domains", "languages", "sentiment_participants", "sentiment_trend"]


def _outputs(b, model_desc: str, synthetic: bool, zip_path: str):
    figs = R.figures(b)
    vals = [R.status_markdown(b, model_desc, synthetic), R.quality_table(b),
            b.participants.reset_index().round(2), R.behaviour_markdown(b), b.behavior["initiation"].round(1),
            b.behavior["no_observed_reply"].round(1), b.text["bigrams"], b.text["distinctive"].round(2),
            R.wordcloud_figure(b), R.sentiment_markdown(b), R.topics_markdown(b), R.keywords_table(b),
            R.stats_table(b), zip_path]
    return vals + [figs.get(k) for k in FIG_KEYS]


def build_demo(settings: AppSettings | None = None) -> gr.Blocks:
    s = settings or AppSettings.from_env()
    feedback = FeedbackStore(s.feedback_path)

    def run(file, anonymize, with_nlp, include_text, progress=gr.Progress()):  # noqa: B008
        if file is None:
            raise gr.Error("Please upload an exported WhatsApp chat (.txt or .zip).")
        try:
            r = process_upload(file, Path(file).name, s, anonymize, with_nlp, include_text, progress)
        except UserFacingError as exc:
            raise gr.Error(str(exc)) from exc
        return _outputs(r.bundle, r.model_description, False, str(r.zip_path))

    def run_demo(progress=gr.Progress()):  # noqa: B008
        text, _ = make_synthetic_chat(3000, seed=42)
        r = process_upload(text.encode("utf-8"), "synthetic.txt", s, True, True, False, progress)
        return _outputs(r.bundle, r.model_description, True, str(r.zip_path))

    def send_feedback(section, rating, comment):
        try:
            feedback.add(section, int(rating), comment or "")
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return "Thank you — feedback saved (no chat content is stored)."

    # delete_cache=(every 5 min, files older than 15 min): Gradio's own copy of uploads/outputs is purged too
    with gr.Blocks(title="chatlens — WhatsApp Chat Intelligence", delete_cache=(300, 900)) as demo:
        gr.Markdown("# 💬 chatlens — WhatsApp Chat Intelligence\nUpload an exported chat "
                    "(*WhatsApp → chat → ⋮ → More → Export chat → Without media*).")
        gr.Markdown(PRIVACY)
        with gr.Row():
            file = gr.File(label="Exported chat (.txt or .zip)", file_types=[".txt", ".zip"], type="filepath")
            with gr.Column():
                anonymize = gr.Checkbox(True, label="Anonymise participant names (recommended)")
                with_nlp = gr.Checkbox(True, label="Run NLP (sentiment, topics, keywords, summaries)")
                include_text = gr.Checkbox(False, label="Include message text in the downloadable dataset "
                                                        "(only possible when names are NOT anonymised)")
                go = gr.Button("Analyse", variant="primary")
                demo_btn = gr.Button("Try with fabricated demo data")
        status = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("Overview"):
                quality = gr.Dataframe(label="Data-quality checks", wrap=True)
                p_overview = gr.Plot(label="Activity heatmap")
            with gr.Tab("Activity"):
                p_hour, p_wd, p_tl = gr.Plot(), gr.Plot(), gr.Plot()
            with gr.Tab("Participants"):
                ptable = gr.Dataframe(label="Participant statistics", wrap=True)
                p_part, p_trend = gr.Plot(), gr.Plot()
            with gr.Tab("Behaviour"):
                beh_md = gr.Markdown()
                p_resp, p_inter = gr.Plot(), gr.Plot()
                init_df = gr.Dataframe(label="Who starts conversations")
                noreply_df = gr.Dataframe(label="Turns with no observed reply (≠ ignored)")
            with gr.Tab("Words & emoji"):
                p_words, p_emoji, p_dom, p_lang = gr.Plot(), gr.Plot(), gr.Plot(), gr.Plot()
                p_cloud = gr.Plot(label="Word cloud (Latin/Devanagari)")
                bigrams = gr.Dataframe(label="Bigrams")
                distinct = gr.Dataframe(label="Participant-specific vocabulary (log-odds z)")
            with gr.Tab("Sentiment"):
                sent_md = gr.Markdown()
                p_sp, p_st = gr.Plot(), gr.Plot()
            with gr.Tab("Topics & keywords"):
                topics_md = gr.Markdown()
                kw_df = gr.Dataframe(label="Keywords by method")
            with gr.Tab("Statistics"):
                stats_df = gr.Dataframe(label="Tests with effect sizes", wrap=True)
                gr.Markdown("Significance ≠ importance: with thousands of messages tiny differences are "
                            "“significant”. Read the *magnitude* column. Messages are not independent, so p-values "
                            "are optimistic.")
            with gr.Tab("Download"):
                dl = gr.File(label="Report + tables + summary (zip)")
            with gr.Tab("Feedback"):
                fb_section = gr.Dropdown(SECTIONS, value="other", label="Which part?")
                fb_rating = gr.Radio([1, 2, 3, 4, 5], value=4, label="How useful was it? (1–5)")
                fb_comment = gr.Textbox(label="Comment (do not paste chat content)", max_lines=4)
                fb_btn, fb_out = gr.Button("Send feedback"), gr.Markdown()
        # Order must match _outputs(): 14 non-figure values, then FIG_KEYS in order.
        outs = [status, quality, ptable, beh_md, init_df, noreply_df, bigrams, distinct, p_cloud, sent_md,
                topics_md, kw_df, stats_df, dl,
                p_overview, p_hour, p_wd, p_tl, p_part, p_trend, p_resp, p_inter, p_words, p_emoji, p_dom, p_lang,
                p_sp, p_st]
        go.click(run, [file, anonymize, with_nlp, include_text], outs)
        demo_btn.click(run_demo, None, outs)
        fb_btn.click(send_feedback, [fb_section, fb_rating, fb_comment], fb_out)
    return demo


def create_app(settings: AppSettings | None = None):
    """FastAPI app: Gradio at ``/`` plus ``/healthz`` for Docker/orchestrators."""
    from fastapi import FastAPI
    s = settings or AppSettings.from_env()
    api = FastAPI(title="chatlens")

    @api.get("/healthz")
    def healthz():
        return {"status": "ok", "work_dir_writable": os.access(tempfile.gettempdir(), os.W_OK)}

    return gr.mount_gradio_app(api, build_demo(s), path="/", max_file_size=f"{s.max_upload_mb}mb")


def main() -> None:
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")),
                log_level="warning")


if __name__ == "__main__":
    main()
