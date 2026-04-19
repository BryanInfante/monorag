# monorag

Sistema de Recuperación Aumentada por Generación (RAG) para documentos técnicos. Indexa archivos PDF y TXT, realiza búsqueda semántica y genera respuestas con un LLM usando el contexto recuperado.

## Stack

- **ChromaDB** — almacenamiento vectorial persistente
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **Groq API** (`llama-3.3-70b-versatile`) — generación de respuestas
- **pdfplumber** — extracción de texto de PDFs
- **Rich** — interfaz CLI interactiva

## Estructura del proyecto

```
monorag/
├── rag_core/              # Paquete principal
│   ├── __init__.py        # Exporta RAGModule
│   ├── module.py          # Orquestador principal (RAGModule)
│   ├── chunker.py         # Fragmentación de texto por tokens
│   ├── embedder.py        # Generación de embeddings
│   ├── retriever.py       # Operaciones con ChromaDB
│   ├── generator.py       # Generación de respuestas vía Groq
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
git clone <url-del-repositorio>
cd monorag

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto con tu clave de API de Groq:

```env
GROQ_API_KEY=tu_clave_aqui
```

Puedes copiar la plantilla incluida:

```bash
cp .env.example .env
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

# Pregunta con respuesta generada
respuesta = rag.ask("¿Cuáles son los tipos de líquidos penetrantes?")
print(respuesta["answer"])
print(respuesta["sources"])

# Listar colecciones
print(rag.list_collections())

# Eliminar colección
rag.delete_collection()
```

## API de RAGModule

| Método                              | Descripción                                      |
|-------------------------------------|--------------------------------------------------|
| `RAGModule(collection)`             | Inicializa con una colección nombrada            |
| `add_documents(directory) -> int`   | Indexa todos los PDF/TXT de un directorio        |
| `add_file(file_path) -> int`        | Indexa un archivo individual (PDF o TXT)         |
| `search(query, top_k=5) -> list`    | Búsqueda semántica, retorna fragmentos           |
| `ask(query, top_k=5) -> dict`       | Pregunta al LLM, retorna respuesta y fuentes     |
| `list_collections() -> list`        | Lista todas las colecciones existentes           |
| `delete_collection()`               | Elimina la colección activa y sus datos          |

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
├── Chunker      → Fragmenta texto en chunks con solapamiento (500 tokens, 50 overlap)
├── Embedder     → Genera embeddings con sentence-transformers
├── Retriever    → Almacena y consulta vectores en ChromaDB
└── Generator    → Genera respuestas con Groq API (llama-3.3-70b-versatile)
```

**Flujo de indexación:** Archivo → Extracción de texto → Fragmentación → Embeddings → ChromaDB

**Flujo de consulta:** Pregunta → Embedding → Búsqueda por similitud → Contexto + Pregunta → LLM → Respuesta

## Licencia

Este proyecto no incluye una licencia explícita. Consulta con el autor antes de redistribuir.
