#!/bin/sh
set -eu

INDEX_DIR="${CITEGROUND_INDEX_DIR:-data/index}"
if [ ! -f "${INDEX_DIR}/manifest.json" ]; then
  echo "No index in ${INDEX_DIR}; ingesting the bundled corpus."
  python -m citeground.cli.ingest
fi

exec uvicorn citeground.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
