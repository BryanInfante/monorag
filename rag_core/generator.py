"""Generator component for LLM-powered answer generation via any OpenAI-compatible API."""

from openai import OpenAI

# System prompt in Spanish for technical normative document Q&A
SYSTEM_PROMPT = (
    "Eres un asistente experto en documentos normativos técnicos. "
    "Responde las preguntas basándote únicamente en el contexto proporcionado. "
    "Si la información no está en el contexto, indícalo claramente. "
    "Responde siempre en español. "
    "No uses notación LaTeX ni fórmulas con $$ o \\frac. Escribe las fórmulas en texto plano (ejemplo: t = PD / (2 × S × E)). "
    "Cita las fuentes utilizadas (nombre del documento y página cuando estén disponibles)."
)


class Generator:
    """Generates answers using any OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", base_url: str | None = None) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            api_key: API key for the LLM provider.
            model: Model identifier for chat completions.
            base_url: Optional base URL for OpenAI-compatible APIs (e.g. Ollama, LM Studio).
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """Generate an answer from query and context chunks.

        Args:
            query: The user's question.
            context_chunks: List of chunk dicts with text and metadata.
            history: Optional list of past conversation turns. Each turn is a
                dict with ``"query"`` and ``"answer"`` keys. Turns are inserted
                as user/assistant message pairs between the system prompt and
                the current user message in chronological order (oldest first).
                When ``None`` or empty, behavior is identical to calling without
                history.

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

        # Build messages list: system prompt, then history turns, then current user message
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["query"]})
                messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error al llamar al LLM: {e}") from e
