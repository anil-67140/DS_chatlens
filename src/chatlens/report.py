"""Self-contained HTML report + data exports.

Security: every string that originates in the chat (names, words, domains, emoji) is HTML-escaped
before it reaches the page; Plotly escapes its own JSON. Exports drop message text by default."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from .analyze import AnalysisBundle, message_level_tests
from .checkpoint import atomic_write_json, atomic_write_text
from .viz import chart_specs, render_plotly

METHODOLOGY = """
<ul>
<li><b>Parsing</b>: every header line becomes exactly one message row; nothing is dropped silently (see counts above).
Day/month order is inferred from the data; if ambiguous it is flagged.</li>
<li><b>Response times</b>: inferred from adjacency (a reply = the first message of a new speaker after another
speaker). Exports contain no reply links or read receipts, so this is an inference. Gaps beyond the response window
are excluded. "No observed reply" does <u>not</u> mean a message was ignored.</li>
<li><b>Sessions</b>: a new session starts after a silence of at least the configured gap.</li>
<li><b>Language</b>: heuristic (script + Roman-Hindi/Urdu word list); Arabic script is labelled Urdu,
Devanagari Hindi.</li>
<li><b>Sentiment/emotion</b>: model predictions only for languages a model is registered for; other messages are
reported as unscored. Accuracy is the benchmark-report accuracy, not 100%.</li>
<li><b>Topics</b>: NMF on conversation sessions, k chosen by NPMI coherence. Coherence is a proxy for interpretability;
read the words.</li>
<li><b>Statistics</b>: non-parametric tests with effect sizes. Messages in a chat are not independent, so p-values are
optimistic; rely on effect sizes and confidence intervals.</li>
</ul>"""


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _table(df: pd.DataFrame, max_rows: int = 30, floatfmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "<p><i>No data.</i></p>"
    d = df.head(max_rows).copy()
    d.index = d.index.map(_esc) if d.index.dtype == object else d.index
    return d.to_html(escape=True, border=0, classes="t", float_format=lambda v: floatfmt.format(v))


def _kv(d: dict) -> str:
    rows = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in d.items())
    return f"<table class='t'>{rows}</table>"


def _test_row(name: str, t: dict | None) -> str:
    if not t:
        return f"<tr><td>{_esc(name)}</td><td colspan=5><i>not computed (insufficient data)</i></td></tr>"
    mag = t.get("magnitude")
    return ("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in (
        name, t["test"], f"{t['p_value']:.3g}", f"{t['effect_size']:.3f} ({t['effect_name']})" if t.get("effect_size")
        is not None else "-", mag or "-", "yes" if t["significant"] else "no")) + "</tr>")


def build_report_html(b: AnalysisBundle, benchmark: dict[str, pd.DataFrame] | None = None,
                      title: str = "WhatsApp Chat Intelligence Report", plotly_js: str = "cdn") -> str:
    ov, rs = b.overview, b.behavior["response_summary"]
    parts = [f"<h1>{_esc(title)}</h1>",
             f"<p class='muted'>chatlens {_esc(b.meta['chatlens_version'])} · {_esc(b.meta['chat_type'])} chat · "
             f"{'anonymised' if b.meta['anonymized'] else 'names shown'} · format {_esc(b.meta['format'])}, "
             f"dates read as {_esc(b.meta['date_order'])}</p>"]
    cards = {"Messages": ov["total_messages"], "Participants": ov["total_participants"], "Words": ov["total_words"],
             "Media": ov["media_count"], "Links": ov["link_count"], "Days": ov["span_days"]}
    card_html = "".join(f"<div class='card'><b>{_esc(f'{v:,}')}</b><span>{_esc(k)}</span></div>"
                        for k, v in cards.items())
    parts.append(f"<div class='cards'>{card_html}</div>")
    pr = {k: v for k, v in b.parse_report.items() if k != "warnings"}
    parts += ["<h2>Parsing and data quality</h2>", _kv(pr)]
    for w in b.parse_report["warnings"]:
        parts.append(f"<p class='warn'>⚠ {_esc(w)}</p>")
    failed = [c for c in b.quality["checks"] if not c["passed"]]
    parts.append("<h3>Quality checks</h3>" + ("<ul>" + "".join(
        f"<li><b>{_esc(c['severity'])}</b> {_esc(c['name'])}: {_esc(c['detail'])}</li>" for c in failed) + "</ul>"
        if failed else "<p>All checks passed.</p>"))
    parts += ["<h2>Participants</h2>", _table(b.participants[["messages", "pct_of_messages", "words", "avg_words",
                                                              "media", "links", "emojis", "active_days"]]),
              "<h3>Participation concentration</h3>", _kv({k: (f"{v:.3f}" if isinstance(v, float) else v)
                                                           for k, v in b.concentration.items()})]
    parts += ["<h2>Conversation behaviour</h2>", f"<p>{_esc(rs.get('counted_responses', 0))} responses counted "
              f"(window {_esc(rs['window_minutes'])} min); {_esc(rs.get('excluded_beyond_window', 0))} excluded.</p>"]
    if "overall" in rs:
        o = rs["overall"]
        parts.append(f"<p>Median response time <b>{o['median_min']:.1f} min</b> "
                     f"(95% CI {o['median_ci95'][0]:.1f}–{o['median_ci95'][1]:.1f}); p90 {o['p90_min']:.1f} min.</p>")
    parts += ["<h3>Who starts conversations</h3>", _table(b.behavior["initiation"]),
              "<h3>Turns with no observed reply (not 'ignored')</h3>", _table(b.behavior["no_observed_reply"])]
    parts.append("<h2>Charts</h2>")
    first = True
    for spec in chart_specs(b).values():
        fig = render_plotly(spec)
        div = fig.to_html(include_plotlyjs=plotly_js if first else False, full_html=False)
        first = False
        note = f"<p class='muted'>{_esc(spec.note)}</p>" if spec.note else ""
        parts.append(f"<section>{div}{note}</section>")
    parts.append("<h2>Text analytics</h2>")
    for key, h in (("top_words", "Top words"), ("bigrams", "Bigrams"), ("trigrams", "Trigrams"), ("emoji", "Emoji"),
                   ("domains", "Domains"), ("distinctive", "Participant-specific vocabulary (log-odds z)")):
        parts += [f"<h3>{h}</h3>", _table(b.text[key], 20)]
    nlp = b.nlp
    parts.append("<h2>NLP</h2>")
    s = nlp.get("sentiment")
    if s:
        parts.append(f"<p>Sentiment: scored {s['n_scored']:,} of {s['n_text_messages']:,} text messages "
                     f"({s['coverage']:.0%}). {_esc(s.get('caveat', ''))}</p>")
        if "by_participant" in s:
            parts.append(_table(s["by_participant"]))
        for a in s.get("monitoring", {}).get("alerts", []):
            parts.append(f"<p class='warn'>⚠ {_esc(a)}</p>")
    t = nlp.get("topics")
    if t and "topics" in t:
        parts.append(f"<h3>Topics (NMF, k={t['k']}, NPMI {t['npmi_coherence']:.2f}, diversity "
                     f"{t['topic_diversity']:.2f}; unit = {_esc(t['unit'])})</h3><ol>" + "".join(
                         f"<li>{_esc(', '.join(w))} <span class='muted'>({n} sessions)</span></li>"
                         for w, n in zip(t["topics"], t["sizes"], strict=True)) + "</ol>")
    elif t:
        parts.append(f"<p><i>Topics skipped: {_esc(t.get('skipped'))}</i></p>")
    kws = nlp.get("keywords", {}).get("keywords")
    if kws:
        parts.append("<h3>Keywords by method</h3><table class='t'>" + "".join(
            f"<tr><td>{_esc(m)}</td><td>{_esc(', '.join(k for k, _ in v[:12]))}</td></tr>" for m, v in kws.items())
            + "</table>")
    parts += ["<h2>Statistical tests</h2><table class='t'><tr><th>comparison</th><th>test</th><th>p</th>"
              "<th>effect size</th><th>magnitude</th><th>p&lt;0.05</th></tr>"]
    tests = message_level_tests(b)
    for k, v in tests["participant_comparison"].items():
        parts.append(_test_row(k, v))
    parts.append(_test_row("weekend vs weekday volume", (tests["weekend_vs_weekday"] or {}).get("test")))
    parts.append(_test_row("response time by responder", tests["response_time_difference"]))
    parts.append(_test_row("sentiment by participant", tests["sentiment_participant_difference"]))
    parts.append("</table><p class='muted'>Significance ≠ importance: with many messages tiny differences become "
                 "significant. Read the magnitude column.</p>")
    if benchmark:
        parts.append("<h2>Model benchmark</h2>")
        for task, tab in benchmark.items():
            parts += [f"<h3>{_esc(task)}</h3>", _table(tab, 40)]
    parts += ["<h2>Methodology and limitations</h2>", METHODOLOGY,
              "<h2>Run configuration</h2>", _kv({k: v for k, v in b.meta["config"].items()}),
              "<h2>Processing time</h2>", _kv({k: f"{v['wall_s']}s, +{v['rss_delta_mb']}MB"
                                               for k, v in b.timings.items()})]
    css = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1f2933}"
           "table.t{border-collapse:collapse;font-size:.9rem;margin:.5rem 0}.t td,.t th{border:1px solid #d9e2ec;"
           "padding:.25rem .5rem}.cards{display:flex;gap:.8rem;flex-wrap:wrap}.card{border:1px solid #d9e2ec;"
           "border-radius:8px;padding:.6rem 1rem;display:flex;flex-direction:column}.card b{font-size:1.4rem}"
           ".muted{color:#627d98}.warn{color:#b44d12}section{margin:1rem 0}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_esc(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")


_TEXT_COLS = ["message", "text_clean", "tokens", "tokens_content", "urls", "emojis"]


def export_bundle(b: AnalysisBundle, out_dir: str | Path, include_text: bool = False,
                  benchmark: dict[str, pd.DataFrame] | None = None) -> dict[str, str]:
    """Write report.html, summary.json and CSV tables. The processed dataset omits message text unless
    ``include_text`` (then you are exporting private content: protect the file)."""
    out = Path(out_dir)
    files: dict[str, str] = {}
    files["report"] = str(atomic_write_text(out / "report.html", build_report_html(b, benchmark)))
    df = b.df if include_text else b.df.drop(columns=[c for c in _TEXT_COLS if c in b.df])
    files["processed_messages"] = str(atomic_write_text(out / "processed_messages.csv",
                                                        df.drop(columns=["tokens", "tokens_content"], errors="ignore")
                                                        .to_csv(index=False)))
    tables = {"participants": b.participants, "initiation": b.behavior["initiation"],
              "sessions": b.behavior["sessions"], "no_observed_reply": b.behavior["no_observed_reply"],
              "top_words": b.text["top_words"], "bigrams": b.text["bigrams"], "emoji": b.text["emoji"],
              "domains": b.text["domains"], "languages": b.text["languages"],
              "heatmap": b.temporal["heatmap"], "monthly": b.temporal["monthly"]}
    for n, t in tables.items():
        if t is not None and len(t):
            files[n] = str(atomic_write_text(out / "tables" / f"{n}.csv", t.to_csv(index=not isinstance(
                t.index, pd.RangeIndex))))
    summary = {"meta": b.meta, "parse_report": b.parse_report, "quality": b.quality,
               "overview": {k: v for k, v in b.overview.items() if not isinstance(v, pd.DataFrame)},
               "concentration": b.concentration, "tests": message_level_tests(b),
               "response_summary": b.behavior["response_summary"], "timings": b.timings,
               "topics": b.nlp.get("topics"), "sentiment": {k: v for k, v in (b.nlp.get("sentiment") or {}).items()
                                                              if not isinstance(v, pd.DataFrame)}}
    files["summary"] = str(atomic_write_json(out / "summary.json", summary))
    json.loads(Path(files["summary"]).read_text())      # self-check: valid JSON
    return files


def project_report(title: str, sections: list[tuple[str, object]], provenance: dict | None = None) -> str:
    """Generic self-contained HTML. A section body may be: a DataFrame (table), a str (escaped paragraph),
    ``("html", str)`` (trusted HTML you generated), or ``("images", [png paths])`` (embedded as base64)."""
    import base64
    parts = [f"<h1>{_esc(title)}</h1>"]
    for head, body in sections:
        parts.append(f"<h2>{_esc(head)}</h2>")
        if isinstance(body, pd.DataFrame):
            parts.append(_table(body, 60))
        elif isinstance(body, str):
            parts.append(f"<p>{_esc(body)}</p>")
        elif isinstance(body, tuple) and body[0] == "html":
            parts.append(body[1])
        elif isinstance(body, tuple) and body[0] == "images":
            for p in body[1]:
                if Path(p).exists():
                    b64 = base64.b64encode(Path(p).read_bytes()).decode()
                    parts.append(f"<img style='max-width:100%' alt='{_esc(Path(p).stem)}' "
                                 f"src='data:image/png;base64,{b64}'>")
        elif isinstance(body, dict):
            parts.append(_kv(body))
    if provenance:
        parts += ["<h2>Provenance</h2>", _kv(provenance)]
    css = ("body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}"
           "table.t{border-collapse:collapse;font-size:.85rem}"
           ".t td,.t th{border:1px solid #d9e2ec;padding:.2rem .5rem}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_esc(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")
