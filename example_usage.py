"""Ejemplo de uso de RAGModule.

Este script demuestra operaciones públicas: instanciación, indexación,
búsqueda semántica/híbrida, preguntas con respuesta generada por LLM, listado
de colecciones y eliminación de colecciones.

Para ejecutar este script necesitás:
  1. Un archivo .env con LLM_API_KEY y, si aplica, LLM_PROVIDER/LLM_BASE_URL/LLM_MODEL.
  2. Documentos PDF, TXT o MD en ./docs (o ajustar las rutas).
"""

from rag_core import RAGModule


def main() -> None:
    """Ejecuta una demostración mínima de RAGModule."""

    print("=== Paso 1: Inicialización ===")
    rag = RAGModule(
        collection="mi_coleccion",
        chunk_size=500,
        chunk_overlap=50,
        # También podés pasar llm_provider, llm_base_url y llm_model acá.
    )
    print("Módulo RAG inicializado correctamente.\n")

    print("=== Paso 2: Indexar directorio ===")
    cantidad_chunks = rag.add_documents("./docs")
    print(f"Se indexaron {cantidad_chunks} fragmentos desde el directorio.\n")

    print("=== Paso 3: Búsqueda semántica/híbrida ===")
    consulta = "¿Cuáles son los requisitos principales?"
    resultados = rag.search(consulta, top_k=5)
    print(f"Se encontraron {len(resultados)} resultados:")
    for i, resultado in enumerate(resultados, start=1):
        fuente = resultado["metadata"]["source"]
        pagina = resultado["metadata"].get("page", "N/A")
        texto_corto = resultado["text"][:120]
        print(f"  {i}. [{fuente}, pág. {pagina}] {texto_corto}...")
    print()

    print("=== Paso 4: Pregunta y respuesta (LLM) ===")
    pregunta = "¿Cuáles son los requisitos principales del documento?"
    respuesta = rag.ask(pregunta)
    print(f"Respuesta: {respuesta['answer']}")
    print(f"Fuentes utilizadas: {len(respuesta['sources'])} fragmentos.\n")

    print("=== Paso 5: Listar colecciones ===")
    colecciones = rag.list_collections()
    print(f"Colecciones disponibles: {colecciones}\n")

    print("=== Paso 6: Eliminar colección ===")
    rag.delete_collection()
    print("Colección eliminada correctamente.\n")


if __name__ == "__main__":
    main()
