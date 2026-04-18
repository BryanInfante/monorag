from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Calls SentenceTransformer.encode() and converts the NumPy array
        output to a list of Python float lists for ChromaDB compatibility.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        embeddings = self.model.encode(texts)
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self.model.encode([query])
        return list(map(float, embedding[0]))
