"""
Text chunking for large documents.
Splits documents into smaller chunks while preserving semantic boundaries.
"""
from typing import List, Optional
from langchain_core.documents import Document


class TextChunker:
    """
    Chunk documents into smaller pieces.

    Uses semantic separators (paragraphs, sentences) when possible,
    with character-based fallback for very long content.
    """

    def __init__(
        self,
        chunk_size: int = 1000,  # Max characters per chunk
        chunk_overlap: int = 200,  # Overlap between chunks
        separators: Optional[List[str]] = None,  # Split on these (priority order)
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, document: Document) -> List[Document]:
        """Split a document into chunks."""
        text = document.page_content

        if len(text) <= self.chunk_size:
            return [document]

        chunks = self._split_text(text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            documents.append(doc)

        return documents

    def chunk_batch(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents."""
        result = []
        for doc in documents:
            result.extend(self.chunk(doc))
        return result

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks using separators."""
        chunks = []
        current_chunk = ""

        # Try to split on paragraph boundaries first
        parts = None
        chosen_separator = " "
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                chosen_separator = separator
                break

        if parts is None:
            # Fallback to character-based splitting
            step = self.chunk_size - self.chunk_overlap
            return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

        for part in parts:
            test_chunk = (
                current_chunk + (chosen_separator if current_chunk else "") + part
            )
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle parts longer than chunk_size
                if len(part) > self.chunk_size:
                    # Split the part itself
                    step = self.chunk_size - self.chunk_overlap
                    for i in range(0, len(part), step):
                        chunks.append(part[i : i + self.chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
