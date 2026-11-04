# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl HOST=0.0.0.0 PORT=7860 \
    CHATLENS_WORK_DIR=/tmp/chatlens_work CHATLENS_FEEDBACK_FILE=/data/feedback.jsonl \
    CHATLENS_MODEL_DIR=/app/models
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown appuser /data
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
COPY configs ./configs
RUN pip install --no-deps .
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).status==200 else 1)"
CMD ["python", "-m", "chatlens.app.gradio_app"]
