# MonoRAG

[English](README.md)

MonoRAG es una capa de conocimiento RAG local-first para convertir tus documentos en contexto reutilizable para asistentes de IA, CLIs, scripts y herramientas compatibles con MCP.

Combina búsqueda híbrida, respuestas con fuentes, storage local persistente, API Python, CLI interactivo y servidor MCP para que tus documentos formen parte de tus flujos de IA sin quedar atados a un producto hosted.

Podés usarlo como:

- CLI interactivo (`monorag`),
- servidor MCP para flujos con agentes (`monorag-mcp`),
- módulo Python reusable (`rag_core`).

## ¿Por qué MonoRAG?

MonoRAG está pensado para desarrolladores y flujos con agentes que necesitan recuperación documental sin atarse por completo a un producto hosted.

- **Listo para MCP**: expone documentos indexados a clientes MCP por STDIO.
- **Agnóstico al proveedor LLM**: funciona con proveedores OpenAI-compatible como OpenAI, Groq, Google AI Studio, Ollama, LM Studio o cualquier endpoint compatible.
- **Vector DB local o remota**: usa ChromaDB local por defecto y soporta despliegues HTTP(S).
- **Configuración persistente del CLI**: LLM, DB, chunking y `top_k` sobreviven reinicios y `pipx upgrade`.
- **Arquitectura reusable**: `RAGModule` acepta inyección de generator, retriever, embedder y chunker.

## Stack

- **ChromaDB** - almacenamiento vectorial local o remoto.
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) - embeddings locales.
- **OpenAI-compatible API** - generación de respuestas con proveedores configurables.
- **pdfplumber** - extracción de texto de PDFs.
- **rank_bm25** - recuperación léxica para búsqueda híbrida.
- **Rich** - interfaz terminal interactiva.
- **FastMCP** - servidor MCP sobre STDIO.

## Arquitectura

`rag_core` es el núcleo reusable. Los adaptadores incluidos usan ChromaDB y un cliente chat OpenAI-compatible, pero el límite público es inyectable.

```text
RAGModule
    Chunker      -> fragmentación por párrafos con overlap
    Embedder     -> embeddings locales por lotes
    Retriever    -> adaptador ChromaDB local/remoto + BM25
    Generator    -> adaptador LLM OpenAI-compatible
```

**Flujo de indexación:** archivo -> extracción -> chunking -> embeddings -> vector DB.

**Flujo de consulta:** pregunta -> embedding -> recuperación híbrida -> contexto + historial -> LLM -> respuesta con fuentes.

## Instalación

### Instalación recomendada con pipx

Instalá MonoRAG directo desde GitHub:

```bash
pipx install "git+https://github.com/BryanInfante/monorag.git"
```

Esto instala dos comandos:

```bash
monorag
monorag-mcp
```

Verificá la versión instalada:

```bash
monorag --version
# o
monorag -V
# o
monorag -v
```

Actualizá una instalación existente:

```bash
pipx upgrade monorag
```

Si `pipx upgrade` no toma el cambio esperado, forzá la reinstalación desde `main`:

```bash
pipx install --force "git+https://github.com/BryanInfante/monorag.git@main"
```

Si todavía no tenés `pipx`:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Después cerrá y abrí la terminal si `ensurepath` lo pide.

### Instalación para desarrollo

Usá este flujo solo si vas a modificar MonoRAG localmente:

```bash
git clone https://github.com/BryanInfante/monorag.git
cd monorag
python -m venv venv
venv\Scripts\activate  # Windows
python -m pip install -e ".[test]"
```

En Linux/macOS:

```bash
source venv/bin/activate
```

## Inicio rápido

```bash
monorag
```

Dentro del CLI:

```text
create soldadura
config llm
index C:\ruta\a\documentos
ask ¿Cuáles son los criterios de aceptación para grietas?
```

`config llm` abre un asistente guiado y guarda la configuración en la carpeta de datos del usuario, así que no tenés que configurar el proveedor cada vez que abrís MonoRAG.

## Configuración LLM

El flujo recomendado desde el CLI es:

```text
config llm
```

MonoRAG te guía por proveedor, base URL, API key y modelo.

También podés configurar con variables de entorno o `.env`:

| Variable | Descripción | Requerida |
| --- | --- | --- |
| `LLM_API_KEY` | Clave del proveedor LLM | Sí, salvo que inyectes un generator custom |
| `LLM_PROVIDER` | Alias incluido: `openai`, `groq`, `google-ai-studio`, `ollama`, `lm-studio` u `openai-compatible` | No |
| `LLM_BASE_URL` | Endpoint OpenAI-compatible; requerido para proveedores custom sin alias | No |
| `LLM_MODEL` | Modelo; si se omite, MonoRAG usa un default por proveedor | No |

Ejemplos:

```env
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Groq
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Google AI Studio mediante endpoint OpenAI-compatible
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
LLM_MODEL=mi-modelo
```

Nota de seguridad: cuando el keyring del sistema operativo está disponible, las API keys configuradas desde el CLI se guardan en el gestor de credenciales del sistema y `config.json` solo conserva una referencia. En entornos headless sin keyring usable, MonoRAG vuelve al archivo local de configuración para mantener el CLI funcional.

## Configuración persistente y storage

MonoRAG guarda la configuración del CLI fuera del entorno del paquete:

- **Windows**: `%LOCALAPPDATA%\monorag\config.json`
- **Linux**: `$XDG_DATA_HOME/monorag/config.json` o `~/.local/share/monorag/config.json`
- **macOS**: `~/Library/Application Support/monorag/config.json`

Podés sobrescribir esa ubicación con:

```env
MONORAG_CONFIG_PATH=C:/ruta/config.json
```

Por defecto, los datos locales de ChromaDB también viven fuera del entorno del paquete:

- **Windows**: `%LOCALAPPDATA%\monorag\chroma_db`
- **Linux**: `$XDG_DATA_HOME/monorag/chroma_db` o `~/.local/share/monorag/chroma_db`
- **macOS**: `~/Library/Application Support/monorag/chroma_db`

Eso significa que los archivos indexados deberían sobrevivir reinicios del CLI y `pipx upgrade monorag`.

Usar una DB local custom:

```env
MONORAG_DB_PATH=C:/data/monorag/chroma_db
```

Usar ChromaDB remoto:

```env
MONORAG_CHROMA_URL=http://localhost:8000
MONORAG_CHROMA_API_KEY=optional-token
MONORAG_CHROMA_TENANT=optional-tenant
MONORAG_CHROMA_DATABASE=optional-database
```

## CLI interactivo

```bash
monorag
```

Comandos principales:

| Comando | Descripción |
| --- | --- |
| `create <nombre>` | Crear y seleccionar una colección |
| `use <nombre>` | Seleccionar una colección existente |
| `index <ruta>` | Indexar archivo o directorio |
| `chat` | Entrar en modo chat con historial en memoria |
| `ask <pregunta>` | Hacer una pregunta puntual |
| `search <consulta>` | Buscar fragmentos relevantes |
| `list` | Listar colecciones |
| `clear` | Limpiar la colección activa |
| `delete` | Eliminar la colección activa |
| `config` | Ver configuración actual |
| `config chunk <size> <overlap> [top_k]` | Guardar chunking y cantidad de resultados opcional |
| `config db path <ruta>` | Guardar ruta local de ChromaDB |
| `config db url <url>` | Guardar URL HTTP(S) de ChromaDB remoto |
| `config db default` | Resetear override de DB |
| `config llm` | Asistente guiado y persistente para LLM |
| `config llm default` | Resetear override persistido de LLM |
| `exit` / `quit` | Salir del CLI |

## Servidor MCP

MonoRAG incluye un servidor MCP para Kiro, Cursor, Claude Desktop o cualquier runtime compatible con MCP.

```bash
monorag-mcp
```

Configuración típica instalada:

```json
{
  "mcpServers": {
    "monorag": {
      "command": "monorag-mcp"
    }
  }
}
```

Configuración para desarrollo desde el repo:

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

El servidor usa STDIO y no abre puertos propios.

Herramientas MCP:

| Tool | Descripción |
| --- | --- |
| `search` | Búsqueda semántica/híbrida |
| `ask` | Respuesta LLM con fuentes |
| `index_file` | Indexar un archivo PDF/TXT/MD |
| `index_directory` | Indexar un directorio |
| `list_collections` | Listar colecciones sin cargar `RAGModule` eagerly |
| `create_collection` | Crear colección |
| `delete_collection` | Eliminar colección |
| `clear_history` | Limpiar historial cacheado de una colección |

## API Python

```python
from rag_core import RAGModule

rag = RAGModule(
    collection="soldadura",
    chunk_size=500,
    chunk_overlap=50,
    llm_api_key="sk-...",
    llm_provider="openai",
    llm_model="gpt-4o",
)

rag.add_documents("./docs")
rag.add_file("./manual.pdf")

resultados = rag.search("criterios de aceptación para grietas", top_k=5)
for resultado in resultados:
    print(resultado["text"][:100], resultado["metadata"])

respuesta = rag.ask("¿Cuáles son los criterios de aceptación para grietas?", top_k=5)
print(respuesta["answer"])
print(respuesta["sources"])
```

### Proveedores y adaptadores custom

Para proveedores OpenAI-compatible, usá `LLM_PROVIDER` + `LLM_BASE_URL` o parámetros del constructor:

```python
rag = RAGModule(
    collection="custom-compatible",
    llm_api_key="token-del-proveedor",
    llm_provider="mi-proveedor",
    llm_base_url="https://llm.example.com/v1",
    llm_model="mi-modelo",
)
```

Para proveedores no compatibles, inyectá un adapter:

```python
from rag_core import RAGModule
from rag_core.generator import Generator

class MiChatProvider:
    def complete(self, *, model, messages):
        return "respuesta desde mi proveedor"

rag = RAGModule(
    collection="custom",
    generator=Generator(
        api_key="token",
        model="modelo-custom",
        provider=MiChatProvider(),
    ),
)
```

También podés inyectar un `generator`, `retriever`, `embedder` o `chunker` completo si necesitás más control.

## Formatos soportados

- **PDF** - extracción página por página con `pdfplumber`.
- **TXT** - texto UTF-8.
- **MD** - Markdown UTF-8.

Los duplicados se detectan por nombre de archivo y se omiten automáticamente.

## Tests

```bash
pytest
```

La suite usa `pytest` y `hypothesis`.

## Licencia

MIT License
