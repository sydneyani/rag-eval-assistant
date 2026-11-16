.PHONY: install ingest test lint format

install:
	python3 -m pip install -e ".[dev,vector]"

ingest:
	python3 -m citeground.cli.ingest

api:
	python3 -m citeground.cli.serve --host 127.0.0.1 --port 8000

ui:
	python3 -m streamlit run ui/streamlit_app.py

test:
	python3 -m pytest

lint:
	python3 -m ruff check .
	python3 -m ruff format --check .

format:
	python3 -m ruff format .
	python3 -m ruff check --fix .
