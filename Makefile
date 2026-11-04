.PHONY: install test lint run docker
install:  ; pip install -e ".[app,tracking,dev]"
test:     ; pytest --cov=chatlens -p no:warnings
lint:     ; ruff check src tests
run:      ; CHATLENS_MODEL_DIR=models python -m chatlens.app.gradio_app
docker:   ; docker compose up --build
