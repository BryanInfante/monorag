# Inspector END — Multi-stage Dockerfile
#
# Stage 1: Download the embedding model (heavy, cached layer)
# Stage 2: Slim runtime with MCP server + web UI

# --- Stage 1: Model download ---
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers==5.4.1

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Install git (needed for pip install from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install MonoRAG from GitHub (brings fastmcp, starlette, chromadb, etc.)
RUN pip install --no-cache-dir "git+https://github.com/BryanInfante/monorag.git"

# Install additional example dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cached embedding model from stage 1
COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface

# Copy example application
COPY config.py .
COPY agent_generator.py .
COPY agents.md .
COPY server.py .
COPY static/ ./static/

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma_db

# Default environment variables
ENV MONORAG_DB_PATH=/app/data/chroma_db
ENV MONORAG_COLLECTION=inspector-end
ENV LLM_PROVIDER=groq
ENV LLM_MODEL=llama-3.3-70b-versatile

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status')" || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
