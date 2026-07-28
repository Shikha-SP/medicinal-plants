#!/bin/bash
set -e

CHROMA_DIR="/app/data/chroma_db"

# Run ingest only if chroma_db is empty or missing — i.e. first deploy.
# On subsequent deploys the volume already has the built database.
if [ ! -d "${CHROMA_DIR}" ] || [ -z "$(ls -A ${CHROMA_DIR} 2>/dev/null)" ]; then
    echo "[entrypoint] chroma_db not found — running ingest_chromadb.py..."
    python scripts/ingest_chromadb.py
    echo "[entrypoint] Ingest complete."
else
    echo "[entrypoint] chroma_db already exists — skipping ingest."
fi

echo "[entrypoint] Starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
