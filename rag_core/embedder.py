import logging
import math

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 256) -> None:
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
        self.model = SentenceTransformer(model_name)
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
            embeddings = self.model.encode(batch)
            result.extend([list(map(float, vec)) for vec in embeddings])

        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self.model.encode([query])
        return list(map(float, embedding[0]))
