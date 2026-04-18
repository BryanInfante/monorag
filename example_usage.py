"""Ejemplo de uso del módulo RAGModule.

Este script demuestra todas las operaciones públicas de RAGModule:
instanciación, indexación de documentos, búsqueda semántica,
preguntas con respuesta generada por LLM, listado de colecciones
y eliminación de colecciones.

Para ejecutar este script necesitas:
  1. Un archivo .env con tu GROQ_API_KEY
  2. Documentos PDF o TXT en un directorio ./docs (o ajustar las rutas)
"""

from rag_core import RAGModule


def main() -> None:
    """Función principal que ejecuta la demostración completa."""

    # --- Paso 1: Crear una instancia de RAGModule con una colección nombrada ---
    print("=== Paso 1: Inicialización ===")
    print("Creando módulo RAG con la colección 'mi_coleccion'...")
    rag = RAGModule(collection="mi_coleccion")
    print("Módulo RAG inicializado correctamente.\n")

    # --- Paso 2: Indexar todos los documentos de un directorio ---
    print("=== Paso 2: Indexar directorio ===")
    print("Indexando todos los archivos PDF y TXT del directorio './docs'...")
    cantidad_chunks = rag.add_documents("./docs")
    print(f"Se indexaron {cantidad_chunks} fragmentos desde el directorio.\n")

    # --- Paso 3: Indexar un archivo individual ---
    print("=== Paso 3: Indexar archivo individual ===")
    print("Indexando el archivo './doc.pdf'...")
    #chunks_archivo = rag.add_file("./doc.pdf")
    #print(f"Se indexaron {chunks_archivo} fragmentos desde el archivo.\n")

    # --- Paso 4: Realizar una búsqueda semántica ---
    print("=== Paso 4: Búsqueda semántica ===")
    consulta = "¿Cuáles son los requisitos principales?"
    print(f"Buscando: '{consulta}' (top_k=5)...")
    resultados = rag.search(consulta, top_k=5)
    print(f"Se encontraron {len(resultados)} resultados:")
    for i, resultado in enumerate(resultados, start=1):
        fuente = resultado["metadata"]["source"]
        pagina = resultado["metadata"]["page"]
        texto_corto = resultado["text"][:120]
        print(f"  {i}. [{fuente}, pág. {pagina}] {texto_corto}...")
    print()

    # --- Paso 5: Hacer una pregunta con respuesta generada por LLM ---
    print("=== Paso 5: Pregunta y respuesta (LLM) ===")
    pregunta = "¿Cuáles son los requisitos principales del documento?"
    print(f"Preguntando: '{pregunta}'...")
    respuesta = rag.ask(pregunta)
    print(f"Respuesta: {respuesta['answer']}")
    print(f"Fuentes utilizadas: {len(respuesta['sources'])} fragmentos.\n")

    # --- Paso 6: Listar todas las colecciones existentes ---
    print("=== Paso 6: Listar colecciones ===")
    colecciones = rag.list_collections()
    print(f"Colecciones disponibles: {colecciones}\n")

    # --- Paso 7: Eliminar la colección y liberar recursos ---
    print("=== Paso 7: Eliminar colección ===")
    print("Eliminando la colección 'mi_coleccion'...")
    rag.delete_collection()
    print("Colección eliminada correctamente.\n")

    print("=== Demostración finalizada ===")


if __name__ == "__main__":
    main()
