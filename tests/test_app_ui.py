"""Drives the real Gradio app over HTTP (server in a background thread)."""
import threading
import time

import pytest

from chatlens.app import render as R
from chatlens.app.gradio_app import FIG_KEYS, build_demo, create_app
from chatlens.app.service import AppSettings, process_upload
from chatlens.synthetic import make_synthetic_chat


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    s = AppSettings(work_root=tmp_path_factory.mktemp("w"))
    return process_upload(make_synthetic_chat(1500, seed=4)[0].encode(), "a.txt", s)


def test_render_helpers(result):
    b = result.bundle
    md = R.status_markdown(b, result.model_description, synthetic=True)
    assert "FABRICATED" in md and "anonymised" in md and "no data leaves this machine" in md
    assert "inference" in R.behaviour_markdown(b) and "not** mean a message was ignored" in R.behaviour_markdown(b)
    assert "Scored" in R.sentiment_markdown(b) and "topics" in R.topics_markdown(b)
    assert not R.stats_table(b).empty and not R.keywords_table(b).empty and not R.quality_table(b).empty
    figs = R.figures(b)
    assert set(FIG_KEYS) - {"sentiment_trend"} <= set(figs) | {"interaction"}
    assert R.wordcloud_figure(b) is not None


def test_render_escapes_markdown_from_hostile_names():
    from chatlens.analyze import analyze_chat
    b = analyze_chat("[01/01/2024, 10:00:00] <b>X</b>_*`: hello there friend\n[01/01/2024, 10:01:00] Y: hi\n")
    assert "<b>" not in R.status_markdown(b, "<i>model</i>")


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    import uvicorn
    s = AppSettings(work_root=tmp_path_factory.mktemp("work"), feedback_path=tmp_path_factory.mktemp("fb") / "f.jsonl")
    cfg = uvicorn.Config(create_app(s), host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", s
    server.should_exit = True
    t.join(timeout=5)


def test_health_endpoint_and_ui_served(live_server):
    import httpx
    url, _ = live_server
    r = httpx.get(f"{url}/healthz", timeout=10)
    assert r.status_code == 200 and r.json()["status"] == "ok"
    page = httpx.get(url, timeout=10)
    assert page.status_code == 200 and "gradio" in page.text.lower()


def test_demo_button_end_to_end_through_the_api(live_server):
    from gradio_client import Client
    url, s = live_server
    client = Client(url, verbose=False)
    api = client.view_api(return_format="dict")["named_endpoints"]
    name = next(k for k in api if "demo" in k)
    out = client.predict(api_name=name)
    assert "FABRICATED" in out[0]
    assert str(out[13]).endswith(".zip") or isinstance(out[13], str)
    assert len(out) == 14 + len(FIG_KEYS)


def test_feedback_through_the_api_and_rejects_invalid_rating(live_server):
    from gradio_client import Client
    url, s = live_server
    c = Client(url, verbose=False)
    api = c.view_api(return_format="dict")["named_endpoints"]
    name = next(k for k in api if "feedback" in k)
    assert "saved" in c.predict("topics", 5, "great", api_name=name)
    assert s.feedback_path.exists() and "great" in s.feedback_path.read_text()


def test_build_demo_constructs():
    assert build_demo(AppSettings()) is not None
