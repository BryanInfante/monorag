"""Generator component for LLM-powered answers.

The Generator builds the RAG prompt and delegates the actual provider call to a
small chat provider adapter. The default adapter speaks OpenAI-compatible chat
completion APIs; non-compatible providers can be injected by passing any object
with ``complete(model=..., messages=...)``.
"""

from rag_core.llm_providers import (
    ChatProvider,
    build_chat_provider,
    default_model_for_provider,
)

# System prompt in Spanish for technical normative document Q&A
SYSTEM_PROMPT = (
    "Eres un asistente experto en documentos normativos t\u00e9cnicos. "
    "Responde las preguntas bas\u00e1ndote \u00fanicamente en el contexto proporcionado. "
    "Si la informaci\u00f3n no est\u00e1 en el contexto, ind\u00edcalo claramente. "
    "Responde siempre en espa\u00f1ol. "
    "No uses notaci\u00f3n LaTeX ni f\u00f3rmulas con $$ o \\frac. "
    "Escribe las f\u00f3rmulas en texto plano (ejemplo: t = PD / (2 \u00d7 S \u00d7 E)). "
    "Cita las fuentes utilizadas (nombre del documento y p\u00e1gina cuando est\u00e9n disponibles)."
)

NON_PAGINATED_SUFFIXES = (".txt", ".md")


def _format_source_reference(metadata: dict) -> str:
    """Format a human-friendly chunk reference for prompts/citations."""
    source = metadata.get("source", "desconocido")
    page = metadata.get("page")
    if isinstance(source, str) and source.lower().endswith(NON_PAGINATED_SUFFIXES):
        if page in (0, None, "", "N/A"):
            return f"fuente: {source}"
    if page in (None, "", "N/A"):
        return f"fuente: {source}"
    return f"fuente: {source}, página: {page}"


class Generator:
    """Generates answers using a configurable LLM chat provider."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        provider_name: str | None = None,
        provider: ChatProvider | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: API key for the LLM provider.
            model: Model identifier for chat completions. If omitted, MonoRAG
                selects a practical default for the provider alias.
            base_url: Optional base URL for OpenAI-compatible APIs.
            provider_name: Built-in provider alias. Examples: ``openai``,
                ``groq``, ``google-ai-studio``, ``ollama``, ``lm-studio``.
            provider: Optional custom chat provider object. Use this for
                providers that do not expose an OpenAI-compatible API.
        """
        self.provider = provider or build_chat_provider(
            provider=provider_name,
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model or default_model_for_provider(provider_name)

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
            RuntimeError: If the LLM provider call fails.
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            context_parts.append(
                f"--- Fragmento {i} ({_format_source_reference(metadata)}) ---\n"
                f"{chunk.get('text', '')}"
            )

        context_text = "\n\n".join(context_parts)
        user_message = f"Contexto:\n{context_text}\n\nPregunta: {query}"

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["query"]})
                messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({"role": "user", "content": user_message})

        try:
            return self.provider.complete(model=self.model, messages=messages)
        except Exception as e:
            raise RuntimeError(f"Error al llamar al LLM: {e}") from e
