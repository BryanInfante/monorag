from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from rag_core.mcp_diagnostics import emit_mcp_breadcrumb
from rag_core.storage_paths import (
    default_chroma_api_key,
    default_chroma_database,
    default_chroma_db_path,
    default_chroma_tenant,
    default_chroma_url,
    parse_chroma_url,
)


class Retriever:
    """Manages ChromaDB collection operations for vector storage and retrieval.

    Supports hybrid search combining semantic (embedding) similarity with
    BM25 keyword matching via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | None = None,
        *,
        remote_url: str | None = None,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize a ChromaDB client and get or create a collection.

        Args:
            collection_name: Name of the collection.
            persist_dir: Path to ChromaDB persistence directory. When omitted,
                defaults to ``MONORAG_DB_PATH`` or the project ``chroma_db``.
            remote_url: Optional HTTP(S) URL for a remote Chroma server. Falls
                back to ``MONORAG_CHROMA_URL`` or ``CHROMA_URL``.
            api_key: Optional API key for hosted/remote Chroma. Falls back to
                ``MONORAG_CHROMA_API_KEY`` or ``CHROMA_API_KEY``.
            tenant: Optional hosted Chroma tenant.
            database: Optional hosted Chroma database.
            headers: Optional explicit HTTP headers for remote Chroma.
        """
        remote_url = remote_url or default_chroma_url()
        api_key = api_key or default_chroma_api_key()
        tenant = tenant or default_chroma_tenant()
        database = database or default_chroma_database()

        if remote_url:
            host, port, ssl = parse_chroma_url(remote_url)
            client_headers = dict(headers or {})
            if api_key and "Authorization" not in client_headers:
                client_headers["Authorization"] = f"Bearer {api_key}"

            emit_mcp_breadcrumb(
                "Retriever:init:before_http_client",
                collection=collection_name,
                detail=f"url={remote_url}",
            )
            client_kwargs = {
                "host": host,
                "port": port,
                "ssl": ssl,
                "headers": client_headers or None,
            }
            if tenant:
                client_kwargs["tenant"] = tenant
            if database:
                client_kwargs["database"] = database
            self._client = chromadb.HttpClient(**client_kwargs)
            emit_mcp_breadcrumb(
                "Retriever:init:after_http_client",
                collection=collection_name,
                detail=f"url={remote_url}",
            )
        elif api_key and tenant and database and hasattr(chromadb, "CloudClient"):
            emit_mcp_breadcrumb(
                "Retriever:init:before_cloud_client",
                collection=collection_name,
                detail=f"tenant={tenant} database={database}",
            )
            self._client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key,
            )
            emit_mcp_breadcrumb(
                "Retriever:init:after_cloud_client",
                collection=collection_name,
                detail=f"tenant={tenant} database={database}",
            )
        else:
            if persist_dir is None:
                persist_dir = default_chroma_db_path()
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            emit_mcp_breadcrumb(
                "Retriever:init:before_persistent_client",
                collection=collection_name,
                detail=f"path={persist_dir}",
            )
            self._client = chromadb.PersistentClient(path=persist_dir)
            emit_mcp_breadcrumb(
                "Retriever:init:after_persistent_client",
                collection=collection_name,
                detail=f"path={persist_dir}",
            )

        self._collection_name = collection_name
        emit_mcp_breadcrumb(
            "Retriever:init:before_get_or_create_collection",
            collection=collection_name,
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )
        emit_mcp_breadcrumb(
            "Retriever:init:after_get_or_create_collection",
            collection=collection_name,
        )

        # Build BM25 index from existing documents in the collection
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_index: BM25Okapi | None = None
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from all documents in the collection."""
        try:
            count = int(self._collection.count())
        except (TypeError, ValueError):
            count = 0

        if count == 0:
            self._bm25_corpus = []
            self._bm25_ids = []
            self._bm25_index = None
            return

        # Fetch all documents from ChromaDB
        all_docs = self._collection.get(include=["documents"])
        documents = all_docs.get("documents", []) if isinstance(all_docs, dict) else []
        ids = all_docs.get("ids", []) if isinstance(all_docs, dict) else []
        self._bm25_corpus = documents or []
        self._bm25_ids = ids or []

        if not self._bm25_corpus:
            self._bm25_index = None
            return

        # Tokenize for BM25 (simple whitespace + lowercase)
        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        if not tokenized or all(len(tokens) == 0 for tokens in tokenized):
            self._bm25_index = None
            return
        self._bm25_index = BM25Okapi(tokenized)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add chunks with embeddings and metadata to the collection.

        Also rebuilds the BM25 index to include the new documents.

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
        # Rebuild BM25 index with the new documents included
        self._rebuild_bm25()

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Query the collection by embedding similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata.
        """
        emit_mcp_breadcrumb("retriever:before_collection_count", collection=self._collection_name)
        count = self._collection.count()
        emit_mcp_breadcrumb(
            "retriever:after_collection_count",
            collection=self._collection_name,
            detail=f"count={count}",
        )

        # Handle empty collection case
        if count == 0:
            emit_mcp_breadcrumb("retriever:return_empty", collection=self._collection_name)
            return []

        emit_mcp_breadcrumb(
            "retriever:before_collection_query",
            collection=self._collection_name,
        )
        results = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        emit_mcp_breadcrumb(
            "retriever:after_collection_query",
            collection=self._collection_name,
        )

        emit_mcp_breadcrumb(
            "retriever:before_output_build",
            collection=self._collection_name,
        )
        output = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for text, metadata in zip(documents, metadatas):
            output.append({"text": text, "metadata": metadata})

        emit_mcp_breadcrumb(
            "retriever:after_output_build",
            collection=self._collection_name,
            detail=f"items={len(output)}",
        )
        emit_mcp_breadcrumb("retriever:return", collection=self._collection_name)
        return output

    def hybrid_query(self, query_text: str, query_embedding: list[float], top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """Hybrid search combining semantic similarity and BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.

        Args:
            query_text: The raw query string for BM25 keyword matching.
            query_embedding: The query embedding vector for semantic search.
            top_k: Number of final results to return.
            rrf_k: RRF constant (default 60). Higher values give more weight
                to lower-ranked results.

        Returns:
            List of dicts with keys: text, metadata.
        """
        count = self._collection.count()
        if count == 0:
            return []

        # Fetch more candidates from each method for better fusion
        candidates_k = min(top_k * 3, count)

        # 1. Semantic search via ChromaDB
        semantic_results = self._collection.query(
            query_embeddings=[query_embedding], n_results=candidates_k
        )
        semantic_ids = semantic_results.get("ids", [[]])[0]

        # 2. BM25 keyword search
        bm25_ids: list[str] = []
        if self._bm25_index is not None and self._bm25_corpus:
            tokenized_query = query_text.lower().split()
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            # Get top candidates_k indices sorted by score descending
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidates_k]
            bm25_ids = [self._bm25_ids[i] for i in top_indices if bm25_scores[i] > 0]

        # 3. Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}

        for rank, doc_id in enumerate(semantic_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score descending, take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        if not sorted_ids:
            return []

        # Fetch full documents and metadata for the final results
        final_results = self._collection.get(ids=sorted_ids, include=["documents", "metadatas"])

        # Build output preserving RRF rank order
        id_to_doc = {}
        for i, doc_id in enumerate(final_results["ids"]):
            id_to_doc[doc_id] = {
                "text": final_results["documents"][i],
                "metadata": final_results["metadatas"][i],
            }

        output = [id_to_doc[doc_id] for doc_id in sorted_ids if doc_id in id_to_doc]
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
