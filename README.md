# monorag

Sistema de Recuperación Aumentada por Generación (RAG) para documentos técnicos. Indexa archivos PDF, TXT y MD, realiza búsqueda híbrida semántica + BM25, y genera respuestas con un LLM usando el contexto recuperado.

## Stack

- **ChromaDB** — almacenamiento vectorial local, remoto por HTTP(S) o alojado.
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) — embeddings locales.
- **OpenAI-compatible API** — generación de respuestas con OpenAI, Groq, Google AI Studio, Ollama, LM Studio u otro endpoint compatible.
- **pdfplumber** — extracción de texto de PDFs.
- **rank_bm25** — búsqueda léxica para recuperación híbrida.
- **Rich** — interfaz CLI interactiva.
- **FastMCP** — servidor MCP sobre STDIO.

## Arquitectura

`rag_core` es el núcleo reusable. El adaptador incluido usa ChromaDB y un cliente OpenAI-compatible, pero `RAGModule` acepta inyección de `generator`, `retriever`, `embedder` y `chunker`, así que el core no queda atado a un proveedor único.

```text
RAGModule
├── Chunker      → fragmentación por párrafos con overlap
├── Embedder     → embeddings locales por lotes
├── Retriever    → adaptador ChromaDB local/remoto + BM25
└── Generator    → adaptador LLM OpenAI-compatible
```

**Flujo de indexación:** archivo → extracción → chunking → embeddings → vector DB.

**Flujo de consulta:** pregunta → embedding → recuperación híbrida → contexto + historial → LLM → respuesta con fuentes.

## Instalación

```bash
git clone https://github.com/BryanInfante/monorag.git
cd monorag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

En Linux/macOS:

```bash
source venv/bin/activate
```

Si querés instalar MonoRAG como herramienta de sistema para tu usuario:

```bash
pip install --user .
```

Eso te deja disponibles dos comandos:

```bash
monorag
monorag-mcp
```

## Configuración LLM

Copia la plantilla:

```bash
cp .env.example .env
```

Variables principales:

| Variable | Descripción | Requerida |
| --- | --- | --- |
| `LLM_API_KEY` | Clave del proveedor LLM | Sí, salvo que inyectes un generator custom |
| `LLM_PROVIDER` | Alias del proveedor incluido: `openai`, `groq`, `google-ai-studio`, `ollama`, `lm-studio` u `openai-compatible` | No |
| `LLM_BASE_URL` | Endpoint OpenAI-compatible; requerido para proveedores custom sin alias | No |
| `LLM_MODEL` | Modelo a usar; si se omite, MonoRAG usa un default práctico por alias | No |

Regla importante: MonoRAG puede hablar por configuración con cualquier proveedor que exponga una API compatible con OpenAI Chat Completions. Para proveedores con API propia/no compatible, inyectá un adapter propio (ver "Inyección de proveedores custom").

Ejemplos:

```env
# OpenAI oficial
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Groq
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Google AI Studio usando endpoint OpenAI-compatible
LLM_PROVIDER=google-ai-studio
LLM_API_KEY=AIza...
LLM_MODEL=gemini-2.0-flash

# Ollama local
LLM_PROVIDER=ollama
LLM_API_KEY=ollama
LLM_MODEL=llama3.2

# Endpoint OpenAI-compatible custom
LLM_PROVIDER=mi-proveedor
LLM_API_KEY=token-del-proveedor
LLM_BASE_URL=https://llm.example.com/v1
LLM_MODEL=modelo-del-proveedor
```

## Configuración de base vectorial

Por defecto MonoRAG usa una carpeta de datos del usuario y crea `chroma_db` automáticamente ahí:

- **Windows**: `%LOCALAPPDATA%\monorag\chroma_db`
- **Linux**: `$XDG_DATA_HOME/monorag/chroma_db` o `~/.local/share/monorag/chroma_db`
- **macOS**: `~/Library/Application Support/monorag/chroma_db`

Si querés, también podés apuntar a cualquier carpeta local:

```env
MONORAG_DB_PATH=C:/data/monorag/chroma_db
```

Para Chroma remoto:

```env
MONORAG_CHROMA_URL=http://localhost:8000
MONORAG_CHROMA_API_KEY=optional-token
MONORAG_CHROMA_TENANT=optional-tenant
MONORAG_CHROMA_DATABASE=optional-database
```

También podés pasar `db_path`, `db_url`, `db_api_key`, `db_tenant` y `db_database` directamente al constructor de `RAGModule`.

## CLI interactivo

```bash
monorag
# o, si estás trabajando desde el repo:
python cli.py
```

Comandos principales:

| Comando | Descripci?n |
| --- | --- |
| `create <nombre>` | Crear y seleccionar una colecci?n |
| `use <nombre>` | Seleccionar una colecci?n existente |
| `index <ruta>` | Indexar archivo o directorio |
| `chat` | Modo chat con historial en memoria |
| `ask <pregunta>` | Pregunta puntual |
| `search <consulta>` | B?squeda de fragmentos relevantes |
| `list` | Listar colecciones |
| `clear` | Limpiar la colecci?n activa |
| `delete` | Eliminar la colecci?n activa |
| `config` | Ver configuraci?n del CLI |
| `config chunk <size> <overlap>` | Configurar chunking para nuevas sesiones |
| `config db path <ruta>` | Usar ChromaDB local en cualquier carpeta |
| `config db url <url>` | Usar ChromaDB remoto HTTP(S) |
| `config db default` | Volver al storage por defecto |
| `config llm` | Abrir asistente guiado para configurar proveedor, base URL, API key y modelo |
| `config llm default` | Volver a `.env`/defaults para LLM |

El flujo recomendado para LLM es `config llm`: el CLI te lleva paso a paso por proveedor, endpoint, clave y modelo. Los shortcuts directos existen para automatizaci?n, pero la UX humana principal es el asistente guiado.

La configuraci?n de `chunk_size` y `chunk_overlap` vive en el CLI o en el c?digo que instancia `RAGModule`; no depende de `.env`.
La configuraci?n LLM del CLI es de sesi?n: se aplica al crear/usar una colecci?n nueva o al volver a seleccionar la colecci?n.

## Uso programático

```python
from rag_core import RAGModule

rag = RAGModule(
    collection="mi_coleccion",
    chunk_size=500,
    chunk_overlap=50,
    llm_api_key="sk-...",
    llm_provider="openai",
    llm_base_url="https://api.openai.com/v1",  # opcional para alias conocidos
    llm_model="gpt-4o",
)

rag.add_documents("./docs")
rag.add_file("./documento.pdf")

resultados = rag.search("¿Qué es un ensayo no destructivo?", top_k=5)
for r in resultados:
    print(r["text"][:100], r["metadata"])

respuesta = rag.ask("¿Cuáles son los tipos de líquidos penetrantes?")
print(respuesta["answer"])
print(respuesta["sources"])
```

### Inyección de proveedores custom

Para proveedores OpenAI-compatible, usá `LLM_PROVIDER` + `LLM_BASE_URL` o los parámetros del constructor:

```python
rag = RAGModule(
    collection="custom-compatible",
    llm_api_key="token-del-proveedor",
    llm_provider="mi-proveedor",
    llm_base_url="https://llm.example.com/v1",
    llm_model="modelo-del-proveedor",
)
```

Si tu proveedor LLM no expone una API OpenAI-compatible, inyectá un provider propio en el `Generator`:

```python
from rag_core.generator import Generator

class MiChatProvider:
    def complete(self, *, model, messages):
        # Acá adaptás la API propia del proveedor al contrato de MonoRAG.
        return "respuesta desde mi proveedor"

rag = RAGModule(
    collection="custom",
    generator=Generator(
        api_key="token",
        model="modelo-propio",
        provider=MiChatProvider(),
    ),
)
```

También podés inyectar un `generator` completo con método `generate(query, context_chunks, history=None)` si necesitás controlar prompt, streaming o herramientas del proveedor.

Si querés otra base vectorial distinta a ChromaDB, inyectá un retriever con los métodos usados por `RAGModule`: `add`, `query` o `hybrid_query`, `has_source`, `delete_collection` y `list_collections`.

## Servidor MCP

MonoRAG incluye un servidor MCP para clientes como Kiro, Cursor o Claude Desktop.

```bash
monorag-mcp
# o
python -m rag_core.mcp_server
```

Configuración típica:

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

El servidor usa STDIO. No abre puertos propios. El arranque es lazy real: no importa `sentence_transformers`, ChromaDB ni OpenAI al importar `rag_core.mcp_server`; esas dependencias cargan cuando una tool necesita una colección.

Herramientas MCP:

| Herramienta | Descripción |
| --- | --- |
| `search` | Búsqueda semántica/híbrida |
| `ask` | Pregunta con respuesta LLM y fuentes |
| `index_file` | Indexar archivo PDF/TXT/MD |
| `index_directory` | Indexar directorio |
| `list_collections` | Listar colecciones sin instanciar `RAGModule` |
| `create_collection` | Crear colección |
| `delete_collection` | Eliminar colección |
| `clear_history` | Limpiar historial de una colección cacheada |

## API de `RAGModule`

| Método | Descripción |
| --- | --- |
| `RAGModule(collection, max_history=10, chunk_size=500, chunk_overlap=50, ...)` | Inicializa una colección |
| `add_documents(directory) -> int` | Indexa PDF/TXT/MD de un directorio |
| `add_file(file_path) -> int` | Indexa un archivo PDF/TXT/MD |
| `search(query, top_k=5) -> list` | Recupera fragmentos relevantes |
| `ask(query, top_k=5) -> dict` | Genera respuesta con fuentes |
| `clear_history()` | Limpia historial de conversación |
| `list_collections() -> list` | Lista colecciones |
| `delete_collection()` | Elimina la colección activa |

## Formatos soportados

- **PDF** — extracción página por página con `pdfplumber`.
- **TXT** — lectura UTF-8 completa.
- **MD** — lectura UTF-8 completa.

Los duplicados se detectan por nombre de archivo y se omiten automáticamente.

## Tests

```bash
pytest
```

El proyecto usa `pytest` y `hypothesis` para pruebas unitarias, de integración y basadas en propiedades.

## Licencia

MIT License
