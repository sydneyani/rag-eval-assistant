.PHONY: install ingest test lint format

install:
	python3 -m pip install -e ".[dev,vector]"

ingest:
	python3 -m citeground.cli.ingest

test:
	python3 -m pytest

lint:
	python3 -m ruff check .
	python3 -m ruff format --check .

format:
	python3 -m ruff format .
	python3 -m ruff check --fix .
