# Inspector END — Example Agent

An example agent built on [MonoRAG](../../README.md) specialized in Non-Destructive Testing (NDT) standards. Demonstrates how to create domain-specific AI assistants using MonoRAG as the knowledge infrastructure.

## Architecture

```
User (Browser) ──► Streamlit App ──► RAGModule
                                        │
                        ┌───────────────┴──────────────┐
                        ▼                              ▼
                 Retriever (ChromaDB)       InspectorGenerator
                 hybrid search             (agents.md personality)
                        │                              │
                        ▼                              ▼
                   chunks ─────────────────────► Groq LLM
                                                       │
                 App renders answer + citations ◄──────┘
```

- **MonoRAG** handles document processing, indexing, embedding, and hybrid retrieval.
- **InspectorGenerator** injects a custom NDT expert personality via `agents.md`.
- **Streamlit** provides a read-only chat interface for end users.
- **Admin** manages documents and configuration through the MonoRAG CLI.

## Quick Start (Local)

### Prerequisites

- Python 3.10+
- A Groq API key ([get one free](https://console.groq.com))

### Setup

```bash
cd examples/inspector-end

# Install MonoRAG from the parent repo
pip install -e ../..

# Install example dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your LLM_API_KEY
```

### Index Documents

Use the MonoRAG CLI to index your NDT documents:

```bash
monorag
```

Inside the CLI:

```
create inspector-end
index /path/to/your/ndt-documents/
exit
```

### Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Docker Deployment

### Build

```bash
docker build -t inspector-end .
```

### Run

```bash
docker run -d \
  --name inspector-end \
  -p 8501:8501 \
  -e LLM_API_KEY=your_groq_api_key \
  -v inspector-end-data:/app/data/chroma_db \
  inspector-end
```

### Index Documents in Docker

```bash
# Open a shell inside the running container
docker exec -it inspector-end bash

# Run MonoRAG CLI
monorag
# > create inspector-end
# > index /path/to/documents
# > exit
```

## OCI Deployment

### 1. Create an OCI Compute Instance

- Image: Oracle Linux 8 or Ubuntu 22.04 (ARM or x86)
- Shape: VM.Standard.A1.Flex (free tier: 4 OCPU, 24GB RAM)
- Open port 8501 in the Security List

### 2. Install Docker

```bash
sudo yum install -y docker   # Oracle Linux
# or
sudo apt install -y docker.io   # Ubuntu

sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

### 3. Deploy

```bash
# Pull or build the image
docker build -t inspector-end .

# Run with your API key
docker run -d \
  --name inspector-end \
  --restart unless-stopped \
  -p 8501:8501 \
  -e LLM_API_KEY=your_groq_api_key \
  -v inspector-end-data:/app/data/chroma_db \
  inspector-end
```

### 4. Index Documents

```bash
docker exec -it inspector-end monorag
# > create inspector-end
# > index /path/to/documents
# > exit
```

The app is now accessible at `http://<OCI_PUBLIC_IP>:8501`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — (required) | Groq API key |
| `LLM_PROVIDER` | `groq` | LLM provider |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model identifier |
| `LLM_BASE_URL` | — | Custom endpoint (optional) |
| `MONORAG_COLLECTION` | `inspector-end` | Collection name |
| `MONORAG_DB_PATH` | OS default | ChromaDB persistence path |

## Customization

To create your own agent for a different domain:

1. Edit `agents.md` with your agent's personality and instructions.
2. Update `SUGGESTED_QUESTIONS` in `config.py` with domain-relevant questions.
3. Update `AGENT_NAME` and `AGENT_DESCRIPTION` in `config.py`.
4. Index your own documents using the MonoRAG CLI.

No code changes required — MonoRAG adapts to any knowledge domain.
