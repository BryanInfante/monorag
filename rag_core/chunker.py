class Chunker:
    """Splits text into overlapping token-based chunks with metadata."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        """Initialize chunker with size and overlap in tokens.

        Args:
            chunk_size: Number of tokens per chunk.
            overlap: Number of overlapping tokens between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str, start_page: int = 0) -> list[dict]:
        """Split text into chunks with metadata.

        Tokenizes text using whitespace splitting, then slides a window of
        chunk_size tokens with step chunk_size - overlap. Each chunk carries
        metadata with source filename, page number, and sequential index.

        Args:
            text: The text to chunk.
            source: Source filename for metadata.
            start_page: Page number for metadata (0 for TXT).

        Returns:
            List of dicts with keys: text, metadata (source, page, chunk_index).
        """
        if not text or not text.strip():
            return []

        tokens = text.split()
        if not tokens:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        i = 0

        while i < len(tokens):
            window = tokens[i : i + self.chunk_size]
            chunks.append({
                "text": " ".join(window),
                "metadata": {
                    "source": source,
                    "page": start_page,
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1
            i += step

        return chunks

    def chunk_pages(self, pages: list[tuple[str, int]], source: str) -> list[dict]:
        """Chunk text from multiple pages, preserving page numbers.

        Concatenates all page texts, tracks which page each token originated
        from, then chunks the combined text and assigns each chunk the page
        number where its first token originated.

        Args:
            pages: List of (page_text, page_number) tuples.
            source: Source filename for metadata.

        Returns:
            List of chunk dicts with page-accurate metadata.
        """
        all_tokens: list[str] = []
        token_page_map: list[int] = []

        for page_text, page_number in pages:
            page_tokens = page_text.split()
            all_tokens.extend(page_tokens)
            token_page_map.extend([page_number] * len(page_tokens))

        if not all_tokens:
            return []

        chunks = []
        step = self.chunk_size - self.overlap
        chunk_index = 0
        i = 0

        while i < len(all_tokens):
            window = all_tokens[i : i + self.chunk_size]
            page = token_page_map[i]
            chunks.append({
                "text": " ".join(window),
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1
            i += step

        return chunks
