# MonoRAG

[Español](README.es.md)

MonoRAG is a local-first RAG knowledge layer for turning your documents into reusable context for AI assistants, CLIs, scripts, and MCP-compatible tools.

It combines hybrid search, source-grounded answers, persistent local storage, a developer-friendly Python API, an interactive CLI, and an MCP server so your documents can become part of your AI workflows without being locked into one hosted product.

Use it as:

- an interactive CLI (`monorag`),
- an MCP server for agent workflows (`monorag-mcp`),
- a reusable Python module (`rag_core`).

## Why MonoRAG?

MonoRAG is built for developers and agent workflows that need document retrieval without locking the whole system to one hosted product.

- **MCP-ready**: expose indexed documents to MCP-compatible clients over STDIO.
- **LLM provider agnostic**: works with OpenAI-compatible providers such as OpenAI, Groq, Google AI Studio, Ollama, LM Studio, or any compatible endpoint.
- **Local or remote vector storage**: uses ChromaDB locally by default, with support for HTTP(S) Chroma deployments.
- **Persistent CLI configuration**: LLM, DB, chunking, and `top_k` settings survive restarts and `pipx upgrade`.
- **Reusable architecture**: `RAGModule` accepts injected generator, retriever, embedder, and chunker implementations.

## Stack

- **ChromaDB** - local or remote vector storage.
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) - local embeddings.
- **OpenAI-compatible API** - answer generation with configurable providers.
- **pdfplumber** - PDF text extraction.
- **rank_bm25** - lexical retrieval for hybrid search.
- **Rich** - interactive terminal UI.
- **FastMCP** - MCP server over STDIO.

## Architecture

`rag_core` is the reusable core. The included adapters use ChromaDB and an OpenAI-compatible chat client, but the public boundary is injectable.

```text
RAGModule
    Chunker      -> paragraph-aware chunking with overlap
    Embedder     -> batched local embeddings
    Retriever    -> ChromaDB local/remote adapter + BM25
    Generator    -> OpenAI-compatible LLM adapter
```

**Indexing flow:** file -> extraction -> chunking -> embeddings -> vector DB.

**Question flow:** question -> embedding -> hybrid retrieval -> context + history -> LLM -> answer with sources.

## Installation

### Recommended install with pipx

Install MonoRAG directly from GitHub:

```bash
pipx install "git+https://github.com/BryanInfante/monorag.git"
```

This installs two commands:

```bash
monorag
monorag-mcp
```

Check the installed version:

```bash
monorag --version
# or
monorag -V
# or
monorag -v
```

Upgrade an existing installation:

```bash
pipx upgrade monorag
```

If `pipx upgrade` does not pick up the expected change, force reinstall from `main`:

```bash
pipx install --force "git+https://github.com/BryanInfante/monorag.git@main"
```

If you do not have `pipx` yet:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Close and reopen your terminal if `ensurepath` asks you to.

### Development install

Use this flow only if you plan to edit MonoRAG locally:

```bash
git clone https://github.com/BryanInfante/monorag.git
cd monorag
python -m venv venv
venv\Scripts\activate  # Windows
python -m pip install -e ".[test]"
```

On Linux/macOS:

```bash
source venv/bin/activate
```

## Quick start

```bash
monorag
```

Inside the CLI:

```text
create welding
config llm
index C:\path\to\documents
ask What are the acceptance criteria for cracks?
```

`config llm` runs a guided setup and saves the configuration in your user data directory, so you do not need to configure the provider every time you open MonoRAG.

## LLM configuration

The recommended CLI flow is:

```text
config llm
```

MonoRAG will guide you through provider, base URL, API key, and model selection.

You can also configure via environment variables or `.env`:

| Variable | Description | Required |
| --- | --- | --- |
| `LLM_API_KEY` | LLM provider API key | Yes, unless you inject a custom generator |
| `LLM_PROVIDER` | Built-in alias: `openai`, `groq`, `google-ai-studio`, `ollama`, `lm-studio`, or `openai-compatible` | No |
| `LLM_BASE_URL` | OpenAI-compatible endpoint; required for custom providers without a built-in alias | No |
| `LLM_MODEL` | Model name; if omitted, MonoRAG picks a provider default | No |

Examples:

```env
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Groq
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Google AI Studio through the OpenAI-compatible endpoint
LLM_PROVIDER=google-ai-studio
LLM_API_KEY=AIza...
LLM_MODEL=gemini-2.0-flash

# Ollama local
LLM_PROVIDER=ollama
LLM_API_KEY=ollama
LLM_MODEL=llama3.2

# Custom OpenAI-compatible endpoint
LLM_PROVIDER=my-provider
LLM_API_KEY=provider-token
LLM_BASE_URL=https://llm.example.com/v1
LLM_MODEL=my-model
```

Security note: when the OS keyring is available, CLI-configured LLM API keys are stored in the system credential manager and `config.json` only keeps a keyring reference. In headless environments without a usable keyring backend, MonoRAG falls back to the local config file so the CLI remains usable.

## Persistent configuration and storage

MonoRAG stores CLI configuration outside the package environment:

- **Windows**: `%LOCALAPPDATA%\monorag\config.json`
- **Linux**: `$XDG_DATA_HOME/monorag/config.json` or `~/.local/share/monorag/config.json`
- **macOS**: `~/Library/Application Support/monorag/config.json`

Override the config path with:

```env
MONORAG_CONFIG_PATH=C:/path/to/config.json
```

By default, local ChromaDB data is stored outside the package environment too:

- **Windows**: `%LOCALAPPDATA%\monorag\chroma_db`
- **Linux**: `$XDG_DATA_HOME/monorag/chroma_db` or `~/.local/share/monorag/chroma_db`
- **macOS**: `~/Library/Application Support/monorag/chroma_db`

That means indexed files should survive CLI restarts and `pipx upgrade monorag`.

Use a custom local DB path:

```env
MONORAG_DB_PATH=C:/data/monorag/chroma_db
```

Use remote ChromaDB:

```env
MONORAG_CHROMA_URL=http://localhost:8000
MONORAG_CHROMA_API_KEY=optional-token
MONORAG_CHROMA_TENANT=optional-tenant
MONORAG_CHROMA_DATABASE=optional-database
```

## Interactive CLI

```bash
monorag
```

Main commands:

| Command | Description |
| --- | --- |
| `create <name>` | Create and select a collection |
| `use <name>` | Select an existing collection |
| `index <path>` | Index a file or directory |
| `chat` | Enter chat mode with in-memory history |
| `ask <question>` | Ask a one-off question |
| `search <query>` | Search relevant chunks |
| `list` | List collections |
| `clear` | Clear the active collection |
| `delete` | Delete the active collection |
| `config` | Show current CLI configuration |
| `config chunk <size> <overlap> [top_k]` | Persist chunking and optional retrieval count |
| `config db path <path>` | Persist a local ChromaDB path |
| `config db url <url>` | Persist a remote ChromaDB HTTP(S) URL |
| `config db default` | Reset persisted DB override |
| `config llm` | Guided and persisted LLM setup |
| `config llm default` | Reset persisted LLM override |
| `exit` / `quit` | Leave the CLI |

## MCP server

MonoRAG includes an MCP server for clients such as Kiro, Cursor, Claude Desktop, or any MCP-compatible agent runtime.

```bash
monorag-mcp
```

Typical installed configuration:

```json
{
  "mcpServers": {
    "monorag": {
      "command": "monorag-mcp"
    }
  }
}
```

Development configuration from a local checkout:

```json
{
  "mcpServers": {
    "monorag": {
      "command": "python",
      "args": ["-m", "rag_core.mcp_server"]
    }
  }
}
```

The server uses STDIO and does not open its own network port.

MCP tools:

| Tool | Description |
| --- | --- |
| `search` | Semantic/hybrid search |
| `ask` | LLM answer with sources |
| `index_file` | Index one PDF/TXT/MD file |
| `index_directory` | Index a directory |
| `list_collections` | List collections without eagerly loading `RAGModule` |
| `create_collection` | Create a collection |
| `delete_collection` | Delete a collection |
| `clear_history` | Clear cached conversation history for a collection |

## Python API

```python
from rag_core import RAGModule

rag = RAGModule(
    collection="welding",
    chunk_size=500,
    chunk_overlap=50,
    llm_api_key="sk-...",
    llm_provider="openai",
    llm_model="gpt-4o",
)

rag.add_documents("./docs")
rag.add_file("./manual.pdf")

results = rag.search("acceptance criteria for cracks", top_k=5)
for result in results:
    print(result["text"][:100], result["metadata"])

answer = rag.ask("What are the acceptance criteria for cracks?", top_k=5)
print(answer["answer"])
print(answer["sources"])
```

### Custom providers and adapters

For OpenAI-compatible providers, use `LLM_PROVIDER` + `LLM_BASE_URL` or constructor parameters:

```python
rag = RAGModule(
    collection="custom-compatible",
    llm_api_key="provider-token",
    llm_provider="my-provider",
    llm_base_url="https://llm.example.com/v1",
    llm_model="my-model",
)
```

For non-compatible providers, inject a provider adapter:

```python
from rag_core import RAGModule
from rag_core.generator import Generator

class MyChatProvider:
    def complete(self, *, model, messages):
        return "answer from my provider"

rag = RAGModule(
    collection="custom",
    generator=Generator(
        api_key="token",
        model="custom-model",
        provider=MyChatProvider(),
    ),
)
```

You can also inject a full `generator`, `retriever`, `embedder`, or `chunker` if you need deeper control.

## Supported formats

- **PDF** - page-by-page extraction with `pdfplumber`.
- **TXT** - UTF-8 text.
- **MD** - UTF-8 Markdown.

Duplicate files are detected by filename and skipped automatically.

## Tests

```bash
pytest
```

The test suite uses `pytest` and `hypothesis`.

## License

MIT License
