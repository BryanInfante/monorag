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

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into non-empty paragraphs on double newline boundaries.

        Splits the input on "\\n\\n", strips whitespace from each fragment,
        and discards any fragments that become empty after stripping.

        Args:
            text: Input text.

        Returns:
            List of trimmed, non-empty paragraph strings.
        """
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _assemble_chunks(self, paragraphs: list[str]) -> list[dict]:
        """Group paragraphs into chunks respecting chunk_size token limit.

        Small paragraphs are accumulated until the next would exceed the limit.
        Large paragraphs (> chunk_size tokens) are split using token-based
        windowing and emitted as individual chunks.

        Args:
            paragraphs: Non-empty paragraph strings.

        Returns:
            List of dicts with keys:
                tokens: flat list of whitespace-delimited tokens in the chunk
                text: chunk text with paragraph boundaries preserved as \\n\\n
        """
        chunks: list[dict] = []
        buffer: list[list[str]] = []
        buffer_token_count = 0

        def flush_buffer() -> None:
            """Flush accumulated paragraph token lists as a single chunk."""
            if not buffer:
                return
            tokens = [t for para_tokens in buffer for t in para_tokens]
            text = "\n\n".join(" ".join(pt) for pt in buffer)
            chunks.append({"tokens": tokens, "text": text})
            buffer.clear()
            nonlocal buffer_token_count
            buffer_token_count = 0

        for paragraph in paragraphs:
            para_tokens = paragraph.split()

            if len(para_tokens) > self.chunk_size:
                # Large paragraph: flush any accumulated small paragraphs first
                flush_buffer()
                # Split with sliding window
                step = max(1, self.chunk_size - self.overlap)
                i = 0
                while i < len(para_tokens):
                    window = para_tokens[i : i + self.chunk_size]
                    chunks.append({
                        "tokens": window,
                        "text": " ".join(window),
                    })
                    i += step
            elif buffer_token_count + len(para_tokens) > self.chunk_size:
                # Adding this paragraph would exceed the limit: flush and start new buffer
                flush_buffer()
                buffer.append(para_tokens)
                buffer_token_count = len(para_tokens)
            else:
                # Accumulate paragraph into current buffer
                buffer.append(para_tokens)
                buffer_token_count += len(para_tokens)

        # Flush any remaining paragraphs
        flush_buffer()

        return chunks

    def chunk(self, text: str, source: str, start_page: int = 0) -> list[dict]:
        """Split text into chunks with metadata using paragraph-aware splitting.

        Splits text on paragraph boundaries (double newlines), accumulates
        small paragraphs into chunks up to chunk_size tokens, and falls back
        to token-based windowing for paragraphs that exceed the limit.
        Overlap tokens from the previous chunk are prepended to each
        subsequent chunk.

        Args:
            text: The text to chunk.
            source: Source filename for metadata.
            start_page: Page number for metadata (0 for TXT).

        Returns:
            List of dicts with keys: text, metadata (source, page, chunk_index).
        """
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        raw_chunks = self._assemble_chunks(paragraphs)
        if not raw_chunks:
            return []

        # Apply overlap and build output dicts
        chunks: list[dict] = []
        prev_tokens: list[str] | None = None

        for i, raw in enumerate(raw_chunks):
            if i == 0:
                final_tokens = raw["tokens"]
                final_text = raw["text"]
            elif self.overlap > 0 and prev_tokens is not None:
                overlap_tokens = prev_tokens[-self.overlap:]
                final_tokens = overlap_tokens + raw["tokens"]
                final_text = " ".join(final_tokens)
            else:
                final_tokens = raw["tokens"]
                final_text = raw["text"]

            prev_tokens = final_tokens

            chunks.append({
                "text": final_text,
                "metadata": {
                    "source": source,
                    "page": start_page,
                    "chunk_index": i,
                },
            })

        return chunks

    def chunk_pages(self, pages: list[tuple[str, int]], source: str) -> list[dict]:
        """Chunk text from multiple pages using paragraph-aware splitting.

        Concatenates all page texts preserving paragraph boundaries, tracks
        which page each token originated from, then uses paragraph-aware
        chunking and assigns each chunk the page number of its first token.

        Args:
            pages: List of (page_text, page_number) tuples.
            source: Source filename for metadata.

        Returns:
            List of chunk dicts with page-accurate metadata.
        """
        if not pages:
            return []

        # Build token_page_map: map each token index to its originating page
        token_page_map: list[int] = []
        for page_text, page_number in pages:
            page_tokens = page_text.split()
            token_page_map.extend([page_number] * len(page_tokens))

        if not token_page_map:
            return []

        # Concatenate page texts with "\n\n" to preserve paragraph boundaries
        combined_text = "\n\n".join(page_text for page_text, _ in pages)

        paragraphs = self._split_paragraphs(combined_text)
        if not paragraphs:
            return []

        raw_chunks = self._assemble_chunks(paragraphs)
        if not raw_chunks:
            return []

        # Compute the global start position of each raw chunk
        raw_start_positions: list[int] = []
        pos = 0
        for raw in raw_chunks:
            raw_start_positions.append(pos)
            pos += len(raw["tokens"])

        # Apply overlap and build output dicts
        chunks: list[dict] = []
        prev_tokens: list[str] | None = None

        for i, raw in enumerate(raw_chunks):
            if i == 0:
                final_tokens = raw["tokens"]
                final_text = raw["text"]
                first_token_pos = 0
            elif self.overlap > 0 and prev_tokens is not None:
                overlap_tokens = prev_tokens[-self.overlap:]
                final_tokens = overlap_tokens + raw["tokens"]
                final_text = " ".join(final_tokens)
                # First token of output chunk is from the overlap prefix
                first_token_pos = max(0, raw_start_positions[i] - self.overlap)
            else:
                final_tokens = raw["tokens"]
                final_text = raw["text"]
                first_token_pos = raw_start_positions[i]

            prev_tokens = final_tokens

            # Look up page number of the first token in this output chunk
            page = token_page_map[first_token_pos] if first_token_pos < len(token_page_map) else token_page_map[-1]

            chunks.append({
                "text": final_text,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_index": i,
                },
            })

        return chunks
