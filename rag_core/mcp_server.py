import logging
import os
import json
from fastmcp import FastMCP
from rag_core.module import RAGModule
import chromadb

# Configuración del servidor MCP
mcp = FastMCP("monorag")
_instances: dict[str, RAGModule] = {}
logger = logging.getLogger(__name__)

def _get_or_create(collection: str) -> RAGModule:
    """Get a cached RAGModule instance or create and cache a new one."""
    if collection not in _instances:
        logger.info(f"Instanciando nuevo RAGModule para la colección: {collection}")
        _instances[collection] = RAGModule(collection=collection)
    return _instances[collection]

@mcp.tool
def search(query: str, collection: str, top_k: int = 5) -> str:
    """Search for relevant document fragments in a collection using semantic similarity."""
    if not query.strip() or not collection.strip():
        return "Error: se requieren parámetros 'query' y 'collection' no vacíos."
    try:
        results = _get_or_create(collection).search(query, top_k)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en search: {e}")
        return f"Error al realizar la búsqueda: {e}"

@mcp.tool
def ask(question: str, collection: str, top_k: int = 5) -> str:
    """Ask a question and get an LLM-generated answer with source references."""
    if not question.strip() or not collection.strip():
        return "Error: se requieren parámetros 'question' y 'collection' no vacíos."
    try:
        answer = _get_or_create(collection).ask(question, top_k)
        return str(answer)
    except Exception as e:
        logger.error(f"Error en ask: {e}")
        return f"Error al responder la pregunta: {e}"

@mcp.tool
def index_file(path: str, collection: str) -> str:
    """Index a single PDF or TXT file into a collection."""
    if not path.strip() or not collection.strip():
        return "Error: se requieren parámetros 'path' y 'collection' no vacíos."
    try:
        count = _get_or_create(collection).add_file(path)
        return f"Archivo indexado correctamente. Fragmentos añadidos: {count}"
    except Exception as e:
        logger.error(f"Error en index_file: {e}")
        return f"Error al indexar el archivo: {e}"

@mcp.tool
def index_directory(path: str, collection: str) -> str:
    """Index all PDF and TXT files from a directory into a collection."""
    if not path.strip() or not collection.strip():
        return "Error: se requieren parámetros 'path' y 'collection' no vacíos."
    try:
        count = _get_or_create(collection).add_documents(path)
        return f"Directorio indexado correctamente. Fragmentos añadidos: {count}"
    except Exception as e:
        logger.error(f"Error en index_directory: {e}")
        return f"Error al indexar el directorio: {e}"

@mcp.tool
def list_collections() -> str:
    """List all available collections in the ChromaDB database."""
    try:
        path = os.getenv("MONORAG_DB_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=path)
        collections = [col.name for col in client.list_collections()]
        return json.dumps(collections, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en list_collections: {e}")
        return f"Error al listar colecciones: {e}"

@mcp.tool
def create_collection(name: str) -> str:
    """Create a new collection and cache its RAGModule instance."""
    if not name.strip():
        return "Error: se requiere el parámetro 'name' no vacío."
    try:
        _get_or_create(name)
        return f"Colección '{name}' creada correctamente."
    except Exception as e:
        logger.error(f"Error en create_collection: {e}")
        return f"Error al crear la colección: {e}"

@mcp.tool
def delete_collection(name: str) -> str:
    """Delete a collection and all its data."""
    if not name.strip():
        return "Error: se requiere el parámetro 'name' no vacío."
    try:
        if name in _instances:
            _instances[name].delete_collection()
            del _instances[name]
        else:
            path = os.getenv("MONORAG_DB_PATH", "./chroma_db")
            client = chromadb.PersistentClient(path=path)
            client.delete_collection(name=name)
        return f"Colección '{name}' eliminada correctamente."
    except Exception as e:
        logger.error(f"Error en delete_collection: {e}")
        return f"Error al eliminar la colección: {e}"

@mcp.tool
def clear_history(collection: str) -> str:
    """Clear conversation history for a cached collection."""
    if not collection.strip():
        return "Error: se requiere el parámetro 'collection' no vacío."
    try:
        if collection in _instances:
            _instances[collection].clear_history()
            return f"Historial de la colección '{collection}' borrado correctamente."
        return f"Error: no existe una sesión activa para la colección '{collection}'."
    except Exception as e:
        logger.error(f"Error en clear_history: {e}")
        return f"Error al borrar el historial: {e}"

def main():
    """Start the MCP server using STDIO transport."""
    mcp.run()

if __name__ == "__main__":
    main()
