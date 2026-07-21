#!/bin/sh
# Start ChromaDB HTTP server in background, then the web app in foreground.

chroma run --host 0.0.0.0 --port 8000 --path /app/data/chroma_db &

exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
