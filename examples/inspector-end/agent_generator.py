"""Custom Generator for the Inspector END agent.

Loads a personality definition from agents.md and injects it as the system
prompt for LLM calls. Implements the generate contract expected by RAGModule.
"""

from pathlib import Path

from rag_core.llm_providers import (
    ChatProvider,
    build_chat_provider,
    default_model_for_provider,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert assistant for Non-Destructive Testing (NDT) standards. "
    "Answer questions using only the provided context. "
    "If the information is not in the context, say so clearly and do not invent anything. "
    "Answer in the same language as the user's question. "
    "Cite the sources used (document name and page when available)."
)

NON_PAGINATED_SUFFIXES = (".txt", ".md")


def _load_agents_prompt(agents_path: str) -> str | None:
    """Load the agents.md file content as a system prompt.

    Returns None if the file does not exist or is empty.
    """
    path = Path(agents_path)
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content if content else None


def _format_source_reference(metadata: dict) -> str:
    """Format a chunk reference for the LLM context."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    if isinstance(source, str) and source.lower().endswith(NON_PAGINATED_SUFFIXES):
        if page in (0, None, "", "N/A"):
            return f"source: {source}"
    if page in (None, "", "N/A"):
        return f"source: {source}"
    return f"source: {source}, page: {page}"


class InspectorGenerator:
    """Generator that uses agents.md as personality for NDT expert answers."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        provider_name: str = "groq",
        base_url: str | None = None,
        agents_path: str | None = None,
        provider: ChatProvider | None = None,
    ) -> None:
        """Initialize the Inspector generator.

        Args:
            api_key: LLM provider API key.
            model: Model identifier. Defaults to provider's default.
            provider_name: Built-in provider alias (default: groq).
            base_url: Optional custom base URL.
            agents_path: Path to agents.md file. Defaults to the bundled one.
            provider: Optional pre-built ChatProvider (useful for testing).
        """
        self.provider = provider or build_chat_provider(
            provider=provider_name,
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model or default_model_for_provider(provider_name)

        if agents_path is None:
            agents_path = str(Path(__file__).parent / "agents.md")

        loaded_prompt = _load_agents_prompt(agents_path)
        self.system_prompt = loaded_prompt or DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """Generate an answer using the NDT expert personality.

        Args:
            query: The user's question.
            context_chunks: Retrieved document chunks with text and metadata.
            history: Optional past conversation turns with query/answer keys.

        Returns:
            Generated answer string.

        Raises:
            RuntimeError: If the LLM provider call fails.
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            context_parts.append(
                f"--- Fragment {i} ({_format_source_reference(metadata)}) ---\n"
                f"{chunk.get('text', '')}"
            )

        context_text = "\n\n".join(context_parts)
        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["query"]})
                messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({"role": "user", "content": user_message})

        try:
            return self.provider.complete(model=self.model, messages=messages)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e
