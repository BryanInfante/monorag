import logging
import math

from rag_core.mcp_diagnostics import emit_mcp_breadcrumb

logger = logging.getLogger(__name__)

# Keep ``rag_core.embedder`` import-light for MCP. Tests patch this module-level
# name directly, so preserve it and lazily populate it when real runtime code
# needs the dependency.
SentenceTransformer = None


def _load_sentence_transformer_class():
    global SentenceTransformer
    if SentenceTransformer is None:
        emit_mcp_breadcrumb("Embedder:init:before_import_sentence_transformer")
        from sentence_transformers import SentenceTransformer as _SentenceTransformer

        SentenceTransformer = _SentenceTransformer
        emit_mcp_breadcrumb("Embedder:init:after_import_sentence_transformer")
    return SentenceTransformer


class Embedder:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 256) -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model.
            batch_size: Maximum number of texts to process in a single
                call to the model's encode() method. Must be >= 1.

        Raises:
            ValueError: If batch_size is less than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"El tamaño de lote debe ser al menos 1, se recibió: {batch_size}"
            )
        emit_mcp_breadcrumb(
            "Embedder:init:before_sentence_transformer",
            detail=f"model={model_name}",
        )
        SentenceTransformerClass = _load_sentence_transformer_class()
        self.model = SentenceTransformerClass(model_name)
        emit_mcp_breadcrumb(
            "Embedder:init:after_sentence_transformer",
            detail=f"model={model_name}",
        )
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Splits the input into batches of at most ``batch_size`` texts and
        calls ``SentenceTransformer.encode()`` once per batch.  The per-batch
        results are concatenated into a single list preserving input order.
        When there is more than one batch, progress is logged at INFO level.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        total_batches = math.ceil(len(texts) / self.batch_size)
        result: list[list[float]] = []

        for b in range(total_batches):
            batch = texts[b * self.batch_size : (b + 1) * self.batch_size]
            if total_batches > 1:
                logger.info("Procesando lote %d de %d", b + 1, total_batches)
            emit_mcp_breadcrumb("embed:before_encode", detail=f"batch={b + 1}/{total_batches}")
            embeddings = self.model.encode(batch)
            emit_mcp_breadcrumb("embed:after_encode", detail=f"batch={b + 1}/{total_batches}")
            result.extend([list(map(float, vec)) for vec in embeddings])

        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        emit_mcp_breadcrumb("embed_query:before_encode")
        embedding = self.model.encode([query])
        emit_mcp_breadcrumb("embed_query:after_encode")
        return list(map(float, embedding[0]))
