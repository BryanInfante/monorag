import chromadb


class Retriever:
    """Manages ChromaDB collection operations for vector storage and retrieval."""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db") -> None:
        """Initialize ChromaDB PersistentClient and get or create collection.

        Args:
            collection_name: Name of the collection.
            persist_dir: Path to ChromaDB persistence directory.
        """
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add chunks with embeddings and metadata to the collection.

        Args:
            ids: Unique IDs for each chunk.
            documents: Chunk text strings.
            embeddings: Embedding vectors.
            metadatas: Metadata dicts for each chunk.
        """
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Query the collection by embedding similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata.
        """
        # Handle empty collection case
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        output = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for text, metadata in zip(documents, metadatas):
            output.append({"text": text, "metadata": metadata})

        return output

    def has_source(self, source: str) -> bool:
        """Check if a source filename already exists in the collection.

        Args:
            source: The source filename to check.

        Returns:
            True if any chunk with this source exists in the collection.
        """
        results = self._collection.get(where={"source": source}, limit=1)
        return len(results["ids"]) > 0

    def delete_collection(self) -> None:
        """Delete the active collection from ChromaDB."""
        self._client.delete_collection(name=self._collection_name)

    def list_collections(self) -> list[str]:
        """List all collection names in the persist directory.

        Returns:
            List of collection name strings.
        """
        collections = self._client.list_collections()
        return [col.name if hasattr(col, "name") else str(col) for col in collections]
