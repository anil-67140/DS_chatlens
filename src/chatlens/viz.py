"""Charts. Each chart is described once (a small spec) and rendered twice: interactive Plotly HTML
and static Matplotlib PNG, so reports work offline and the dashboard stays interactive.

Word clouds: Latin and Devanagari render with a suitable font; Arabic-script (Urdu) needs text
shaping that the wordcloud library does not do, so Urdu words are shown in the bar charts instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402


@dataclass
class ChartSpec:
    name: str
    kind: str                       # bar | barh | line | heatmap | stacked_bar
    title: str
    x: list = field(default_factory=list)
    series: dict[str, list] = field(default_factory=dict)   # name -> y values (or heatmap rows)
    xlabel: str = ""
    ylabel: str = ""
    note: str = ""


def render_plotly(spec: ChartSpec) -> go.Figure:
    fig = go.Figure()
    if spec.kind == "heatmap":
        z = np.array(list(spec.series.values()))
        fig.add_trace(go.Heatmap(z=z, x=spec.x, y=list(spec.series), colorscale="Blues"))
    elif spec.kind in ("bar", "stacked_bar"):
        for n, y in spec.series.items():
            fig.add_trace(go.Bar(x=spec.x, y=y, name=n))
        fig.update_layout(barmode="stack" if spec.kind == "stacked_bar" else "group")
    elif spec.kind == "barh":
        for n, y in spec.series.items():
            fig.add_trace(go.Bar(y=spec.x, x=y, name=n, orientation="h"))
        fig.update_yaxes(autorange="reversed")
    elif spec.kind == "line":
        for n, y in spec.series.items():
            fig.add_trace(go.Scatter(x=spec.x, y=y, name=n, mode="lines"))
    else:
        raise ValueError(spec.kind)
    fig.update_layout(title=spec.title, xaxis_title=spec.xlabel, yaxis_title=spec.ylabel, template="plotly_white",
                      margin=dict(l=40, r=20, t=60, b=40), showlegend=len(spec.series) > 1 and spec.kind != "heatmap")
    return fig


def render_mpl(spec: ChartSpec, figsize=(8, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    names = list(spec.series)
    if spec.kind == "heatmap":
        z = np.array(list(spec.series.values()))
        im = ax.imshow(z, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(spec.x)), [str(v) for v in spec.x], fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    elif spec.kind in ("bar", "stacked_bar"):
        xs = np.arange(len(spec.x))
        w = 0.8 if spec.kind == "stacked_bar" else 0.8 / max(len(names), 1)
        bottom = np.zeros(len(xs))
        for i, n in enumerate(names):
            y = np.asarray(spec.series[n], float)
            if spec.kind == "stacked_bar":
                ax.bar(xs, y, w, bottom=bottom, label=n)
                bottom += y
            else:
                ax.bar(xs + i * w, y, w, label=n)
        many = len(xs) > 8
        ax.set_xticks(xs + (0 if spec.kind == "stacked_bar" else w * (len(names) - 1) / 2),
                      [str(v) for v in spec.x], rotation=45 if many else 0, ha="right" if many else "center",
                      fontsize=8)
    elif spec.kind == "barh":
        y = np.arange(len(spec.x))
        ax.barh(y, list(spec.series.values())[0])
        ax.set_yticks(y, [str(v) for v in spec.x], fontsize=8)
        ax.invert_yaxis()
    elif spec.kind == "line":
        for n in names:
            ax.plot(spec.x, spec.series[n], label=n)
        fig.autofmt_xdate()
    ax.set_title(spec.title, fontsize=11)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    if len(names) > 1 and spec.kind != "heatmap":
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _s(series: pd.Series) -> list:
    return [float(v) for v in series.to_numpy()]


def chart_specs(b) -> dict[str, ChartSpec]:
    """All dashboard charts for an AnalysisBundle. Charts for data that does not exist are omitted."""
    t, beh, tx = b.temporal, b.behavior, b.text
    sp: dict[str, ChartSpec] = {}
    hm = t["heatmap"]
    if hm.to_numpy().sum():
        sp["activity_heatmap"] = ChartSpec("activity_heatmap", "heatmap", "Messages by weekday and hour",
                                           list(hm.columns), {d: _s(hm.loc[d]) for d in hm.index}, "Hour of day")
    if t["hourly"].sum():
        sp["hourly"] = ChartSpec("hourly", "bar", "Messages by hour of day", list(t["hourly"].index),
                                 {"messages": _s(t["hourly"])}, "Hour", "Messages")
        sp["weekday"] = ChartSpec("weekday", "bar", "Messages by weekday", list(t["weekday"].index),
                                  {"messages": _s(t["weekday"])}, "", "Messages")
    g = b.overview.get("growth_daily")
    if g is not None and len(g):
        sp["timeline"] = ChartSpec("timeline", "line", "Conversation volume over time", list(g["period"]),
                                   {"messages per day": _s(g["messages"])}, "Date", "Messages")
        sp["growth"] = ChartSpec("growth", "line", "Cumulative messages", list(g["period"]),
                                 {"cumulative": _s(g["cumulative"])}, "Date", "Messages")
    p = b.participants
    if len(p):
        top = p.head(15)
        sp["participants"] = ChartSpec("participants", "barh", "Messages by participant", list(top.index),
                                       {"messages": _s(top["messages"])}, "Messages")
        sp["participant_words"] = ChartSpec("participant_words", "barh", "Average words per message",
                                            list(top.index), {"avg words": _s(top["avg_words"])}, "Words")
    ma = participants_monthly(b)
    if ma is not None:
        sp["participant_trend"] = ChartSpec("participant_trend", "line", "Monthly messages by participant",
                                            list(ma.index), {c: _s(ma[c]) for c in ma.columns}, "Month", "Messages")
    rd = beh["response_distribution"]
    if len(rd) and rd["responses"].sum():
        sp["response_times"] = ChartSpec("response_times", "bar", "Response-time distribution (within window)",
                                         list(rd["bucket"]), {"responses": _s(rd["responses"])}, "Time to reply",
                                         "Replies", "Inferred from message adjacency; see methodology.")
    im = beh["interaction_matrix"]
    if im.shape[0] > 1 and im.to_numpy().sum():
        sp["interaction"] = ChartSpec("interaction", "heatmap", "Who replies after whom (row replies after column)",
                                      list(im.columns), {r: _s(im.loc[r]) for r in im.index})
    for key, title, col in (("top_words", "Most common words", "word"), ("emoji", "Most used emoji", "emoji"),
                            ("domains", "Most shared domains", "domain"), ("bigrams", "Most common bigrams", "ngram")):
        d = tx.get(key)
        if d is not None and len(d):
            sp[key] = ChartSpec(key, "barh", title, list(d[col].head(15)), {"count": _s(d["count"].head(15))}, "Count")
    lg = tx.get("languages")
    if lg is not None and len(lg):
        sp["languages"] = ChartSpec("languages", "bar", "Detected language of messages (heuristic)",
                                    list(lg["language"]), {"messages": _s(lg["messages"])})
    ld = tx.get("length")
    if ld:
        sp["length"] = ChartSpec("length", "bar", "Message length (words)", list(ld["histogram"]["words"]),
                                 {"messages": _s(ld["histogram"]["messages"])}, "Words", "Messages")
    s = b.nlp.get("sentiment")
    if s and "by_participant" in s:
        bp = s["by_participant"].head(12)
        cols = [c for c in ("pct_positive", "pct_neutral", "pct_negative") if c in bp]
        sp["sentiment_participants"] = ChartSpec("sentiment_participants", "stacked_bar",
                                                 "Predicted sentiment by participant (% of scored messages)",
                                                 list(bp.index), {c[4:]: _s(bp[c]) for c in cols}, "", "%",
                                                 f"Coverage {s['coverage']:.0%}; model predictions, not ground truth.")
        if "monthly_polarity" in s:
            mp = s["monthly_polarity"]
            sp["sentiment_trend"] = ChartSpec("sentiment_trend", "line", "Mean predicted polarity by month",
                                              list(mp.index), {"mean polarity": _s(mp["mean_polarity"])}, "Month",
                                              "-1 negative .. +1 positive")
    tp = b.nlp.get("topics")
    if tp and "topics" in tp:
        sp["topic_sizes"] = ChartSpec("topic_sizes", "bar", f"Sessions per topic (k={tp['k']})",
                                      [f"T{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(tp["topics"])],
                                      {"sessions": [float(v) for v in tp["sizes"]]})
    return sp


def participants_monthly(b) -> pd.DataFrame | None:
    from .analytics import participants as pa
    ma = pa.monthly_activity(b.df)
    return ma if len(ma) >= 2 else None


def make_wordcloud(freq: dict[str, float], font_path: str | None = None):
    """Matplotlib figure with a word cloud (Latin/Devanagari). Returns None for empty input."""
    if not freq:
        return None
    from wordcloud import WordCloud
    wc = WordCloud(width=900, height=450, background_color="white", font_path=font_path,
                   max_words=100, random_state=42).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def save_all(b, out_dir, interactive: bool = True, static: bool = True) -> dict[str, list[str]]:
    """Write every chart (HTML + PNG) atomically. Returns the file lists."""
    from pathlib import Path

    from .checkpoint import atomic_write_bytes, atomic_write_text
    out = Path(out_dir)
    written: dict[str, list[str]] = {"html": [], "png": []}
    for name, spec in chart_specs(b).items():
        if interactive:
            f = atomic_write_text(out / "html" / f"{name}.html", render_plotly(spec).to_html(include_plotlyjs="cdn"))
            written["html"].append(str(f))
        if static:
            import io
            fig = render_mpl(spec)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=140)
            plt.close(fig)
            written["png"].append(str(atomic_write_bytes(out / "png" / f"{name}.png", buf.getvalue())))
    return written


def confusion_figure(metrics: dict, normalize: bool = True, title: str = ""):
    cm = np.array(metrics["confusion_matrix"]["matrix"], float)
    labels = metrics["confusion_matrix"]["labels"]
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{shown[i, j]:.0%}" if normalize else f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if shown[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (row-normalised)" if normalize else title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def benchmark_figure(table, metric: str = "macro_f1", title: str = ""):
    """Horizontal bars of ``metric`` with 95% CI whiskers (needs macro_f1_ci_low/high columns)."""
    t = table.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t[metric], color="#4c78a8")
    if {"macro_f1_ci_low", "macro_f1_ci_high"} <= set(t.columns) and metric == "macro_f1":
        ax.errorbar(t[metric], y, xerr=[t[metric] - t["macro_f1_ci_low"], t["macro_f1_ci_high"] - t[metric]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_yticks(y, t["method"], fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by method (95% bootstrap CI)")
    fig.tight_layout()
    return fig
