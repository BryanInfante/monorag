# Inspector END — Multi-stage Dockerfile
#
# Stage 1: Download the embedding model (heavy, cached layer)
# Stage 2: Slim runtime with Streamlit app

# --- Stage 1: Model download ---
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers==5.4.1

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Install MonoRAG and example dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MonoRAG from GitHub
RUN pip install --no-cache-dir "git+https://github.com/BryanInfante/monorag.git"

# Copy cached embedding model from stage 1
COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface

# Copy example application
COPY config.py .
COPY agent_generator.py .
COPY agents.md .
COPY app.py .

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma_db

# Default environment variables
ENV MONORAG_DB_PATH=/app/data/chroma_db
ENV MONORAG_COLLECTION=inspector-end
ENV LLM_PROVIDER=groq
ENV LLM_MODEL=llama-3.3-70b-versatile
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
