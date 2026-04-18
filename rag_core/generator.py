"""Generator component for LLM-powered answer generation via Groq API."""

from groq import Groq

# System prompt in Spanish for technical normative document Q&A
SYSTEM_PROMPT = (
    "Eres un asistente experto en documentos normativos técnicos. "
    "Responde las preguntas basándote únicamente en el contexto proporcionado. "
    "Si la información no está en el contexto, indícalo claramente. "
    "Responde siempre en español. "
    "Cita las fuentes utilizadas (nombre del documento y página cuando estén disponibles)."
)


class Generator:
    """Generates answers using the Groq API."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        """Initialize the Groq client.

        Args:
            api_key: Groq API key.
            model: Model identifier for chat completions.
        """
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer from query and context chunks.

        Args:
            query: The user's question.
            context_chunks: List of chunk dicts with text and metadata.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If the Groq API call fails.
        """
        # Build user message with context chunks and the question
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "desconocido")
            page = metadata.get("page", "N/A")
            context_parts.append(
                f"--- Fragmento {i} (fuente: {source}, página: {page}) ---\n"
                f"{chunk.get('text', '')}"
            )

        context_text = "\n\n".join(context_parts)
        user_message = (
            f"Contexto:\n{context_text}\n\n"
            f"Pregunta: {query}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error de la API de Groq: {e}") from e
