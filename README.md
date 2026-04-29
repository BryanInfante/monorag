# monorag

Sistema de Recuperación Aumentada por Generación (RAG) para documentos técnicos. Indexa archivos PDF y TXT, realiza búsqueda semántica y genera respuestas con un LLM usando el contexto recuperado.

## Stack

- **ChromaDB** — almacenamiento vectorial persistente
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **OpenAI-compatible API** — generación de respuestas (Groq, OpenAI, Ollama, LM Studio, etc.)
- **pdfplumber** — extracción de texto de PDFs
- **Rich** — interfaz CLI interactiva

## Estructura del proyecto

```
monorag/
├── rag_core/              # Paquete principal
│   ├── __init__.py        # Exporta RAGModule
│   ├── module.py          # Orquestador principal (RAGModule)
│   ├── chunker.py         # Fragmentación inteligente por párrafos
│   ├── embedder.py        # Generación de embeddings por lotes
│   ├── retriever.py       # Operaciones con ChromaDB
│   ├── generator.py       # Generación de respuestas vía Groq (con historial)
│   └── utils.py           # Extracción de texto (PDF, TXT)
├── tests/                 # Tests unitarios y property-based
├── docs/                  # Documentos para indexar
├── chroma_db/             # Base de datos vectorial (persistente)
├── cli.py                 # CLI interactivo (REPL)
├── example_usage.py       # Ejemplo de uso programático
├── requirements.txt       # Dependencias
├── .env.example           # Plantilla de variables de entorno
└── README.md
```

## Instalación

```bash
# Clonar el repositorio
git clone [<url-del-repositorio>](https://github.com/BryanInfante/monorag.git)
cd monorag

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto. Copia la plantilla incluida:

```bash
cp .env.example .env
```

Variables disponibles:

| Variable       | Descripción                                      | Requerida |
|----------------|--------------------------------------------------|-----------|
| `LLM_API_KEY`  | Clave de API del proveedor LLM                   | Sí        |
| `LLM_BASE_URL` | URL base del endpoint (Groq, Ollama, LM Studio…) | No        |
| `LLM_MODEL`    | Nombre del modelo a usar                         | No        |

Ejemplos por proveedor:

```env
# Groq
LLM_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile

# OpenAI
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Ollama (local)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

## Uso

### CLI interactivo

```bash
python cli.py
```

Comandos disponibles:

| Comando            | Descripción                                    |
|--------------------|------------------------------------------------|
| `create <nombre>`  | Crear y seleccionar una colección              |
| `use <nombre>`     | Seleccionar una colección existente            |
| `index <ruta>`     | Indexar un archivo o directorio                |
| `chat`             | Modo chat — escribe preguntas directamente     |
| `ask <pregunta>`   | Hacer una pregunta puntual                     |
| `search <consulta>`| Buscar fragmentos relevantes                   |
| `list`             | Listar todas las colecciones                   |
| `clear`            | Limpiar documentos de la colección activa      |
| `delete`           | Eliminar la colección activa                   |
| `exit`             | Salir del CLI                                  |

### Uso programático

```python
from rag_core import RAGModule

# Crear una instancia con una colección
rag = RAGModule(collection="mi_coleccion")

# Indexar documentos
rag.add_documents("./docs")          # Directorio completo
rag.add_file("./documento.pdf")      # Archivo individual

# Búsqueda semántica
resultados = rag.search("¿Qué es un ensayo no destructivo?", top_k=5)
for r in resultados:
    print(r["text"][:100], r["metadata"])

# Pregunta con respuesta generada (acumula historial automáticamente)
respuesta = rag.ask("¿Cuáles son los tipos de líquidos penetrantes?")
print(respuesta["answer"])
print(respuesta["sources"])

# Pregunta de seguimiento (el LLM recuerda la conversación anterior)
respuesta = rag.ask("¿Puedes dar más detalles sobre el primero?")
print(respuesta["answer"])

# Limpiar historial de conversación
rag.clear_history()

# Listar colecciones
print(rag.list_collections())

# Eliminar colección
rag.delete_collection()
```

### Historial de conversación

`RAGModule` mantiene un historial en memoria de las preguntas y respuestas anteriores. Cada llamada a `ask()` envía los últimos N turnos al LLM para que pueda responder preguntas de seguimiento con contexto.

```python
# Configurar el número máximo de turnos (por defecto 10)
rag = RAGModule(collection="mi_coleccion", max_history=5)

# Cada ask() acumula historial automáticamente
rag.ask("¿Qué dice el documento sobre seguridad?")
rag.ask("¿Y sobre los procedimientos de emergencia?")  # El LLM recuerda la pregunta anterior

# Limpiar historial para empezar un tema nuevo
rag.clear_history()

# Desactivar historial completamente
rag_sin_historial = RAGModule(collection="otra", max_history=0)
```

## API de RAGModule

| Método                                          | Descripción                                      |
|-------------------------------------------------|--------------------------------------------------|
| `RAGModule(collection, max_history=10, llm_api_key=None, llm_base_url=None, llm_model=None)` | Inicializa con colección, historial y configuración LLM |
| `add_documents(directory) -> int`               | Indexa todos los PDF/TXT de un directorio        |
| `add_file(file_path) -> int`                    | Indexa un archivo individual (PDF o TXT)         |
| `search(query, top_k=5) -> list`                | Búsqueda semántica, retorna fragmentos           |
| `ask(query, top_k=5) -> dict`                   | Pregunta al LLM con historial, retorna respuesta |
| `clear_history()`                               | Limpia el historial de conversación              |
| `list_collections() -> list`                    | Lista todas las colecciones existentes           |
| `delete_collection()`                           | Elimina la colección activa y sus datos          |

## Formatos soportados

- **PDF** — extracción página por página con pdfplumber
- **TXT** — lectura completa con codificación UTF-8

Los archivos duplicados (mismo nombre de archivo) se omiten automáticamente al indexar.

## Tests

```bash
pytest
```

El proyecto usa **pytest** para tests unitarios y **hypothesis** para tests basados en propiedades.

## Arquitectura interna

```
RAGModule (orquestador)
├── Chunker      → Fragmentación inteligente por párrafos (500 tokens, 50 overlap)
├── Embedder     → Embeddings por lotes configurables (batch_size=256)
├── Retriever    → Almacena y consulta vectores en ChromaDB
└── Generator    → Respuestas vía API compatible con OpenAI con historial de conversación
```

**Flujo de indexación:** Archivo → Extracción de texto → Fragmentación por párrafos → Embeddings por lotes → ChromaDB

**Flujo de consulta:** Pregunta → Embedding → Búsqueda por similitud → Historial + Contexto + Pregunta → LLM → Respuesta

### Chunking inteligente

El Chunker respeta los límites de párrafo (doble salto de línea `\n\n`) al fragmentar texto. Los párrafos pequeños se acumulan en un solo chunk hasta alcanzar el límite de tokens. Los párrafos que exceden el límite se dividen con ventana deslizante como fallback. Esto produce chunks más coherentes para el LLM.

### Batch embeddings

El Embedder procesa textos en lotes configurables (por defecto 256) en lugar de todos a la vez. Esto reduce el consumo de memoria al indexar grandes volúmenes de documentos. El progreso se registra a nivel INFO cuando hay múltiples lotes.

### Historial de conversación

El Generator recibe los últimos N turnos de conversación (configurable via `max_history`) como mensajes user/assistant entre el system prompt y la pregunta actual. Esto permite al LLM mantener contexto entre preguntas consecutivas.

## Licencia

Este proyecto no incluye una licencia explícita. Consulta con el autor antes de redistribuir.
