"""
Unified document importer for Vector Graph RAG.

Supports importing from:
- Local files: PDF, DOCX, TXT, MD, HTML
- URLs: Web pages (fetched and converted)

Focus: Text documents only.
"""
from typing import List
from pathlib import Path
from pydantic import BaseModel
from langchain_core.documents import Document

from .converter import DocumentConverter, ConversionResult
from .url_fetcher import URLFetcher
from .chunker import TextChunker


class LoaderResult(BaseModel):
    """Result of document loading."""

    documents: List[Document]
    errors: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class DocumentImporter:
    """
    Unified document importer for text documents.

    Supported formats:
    - PDF (via MarkItDown)
    - DOCX (via MarkItDown)
    - URLs (via trafilatura)
    - TXT, MD, HTML (passthrough)

    Example:
        importer = DocumentImporter(chunk_documents=True)
        result = importer.import_sources([
            "/path/to/document.pdf",
            "/path/to/report.docx",
            "https://example.com/article",
        ])
        for doc in result.documents:
            print(doc.page_content)
    """

    # Supported file extensions (text only)
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".md",
        ".html",
        ".htm",
    }

    def __init__(
        self,
        chunk_documents: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.converter = DocumentConverter()
        self.url_fetcher = URLFetcher()
        self.chunker = (
            TextChunker(chunk_size, chunk_overlap) if chunk_documents else None
        )

    def import_sources(
        self,
        sources: List[str],
    ) -> LoaderResult:
        """
        Import documents from multiple sources.

        Args:
            sources: List of file paths or URLs

        Returns:
            LoaderResult with Documents and any errors
        """
        all_documents = []
        all_errors = []

        for source in sources:
            result = self._import_single(source)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        # Apply chunking if enabled
        if self.chunker and all_documents:
            all_documents = self.chunker.chunk_batch(all_documents)

        return LoaderResult(documents=all_documents, errors=all_errors)

    def _import_single(self, source: str) -> ConversionResult:
        """Import a single source (file or URL)."""
        # Check if URL
        if source.startswith(("http://", "https://")):
            return self.url_fetcher.fetch(source)

        # Check if file exists
        path = Path(source)
        if not path.exists():
            return ConversionResult(documents=[], errors=[f"File not found: {source}"])

        # Check if supported
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return ConversionResult(
                documents=[], errors=[f"Unsupported file type: {ext}"]
            )

        # Handle different file types
        if ext in (".pdf", ".docx", ".doc"):
            # Use MarkItDown for PDF and DOCX
            return self.converter.convert(source)
        elif ext in (".txt", ".md", ".html", ".htm"):
            # Direct passthrough for text files
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "source_type": ext[1:],  # Remove dot
                    },
                )
                return ConversionResult(documents=[doc])
            except Exception as e:
                return ConversionResult(
                    documents=[], errors=[f"Failed to read {source}: {str(e)}"]
                )
        else:
            return ConversionResult(
                documents=[], errors=[f"Unsupported file type: {ext}"]
            )

    def import_text(self, text: str, source: str = "text_input") -> LoaderResult:
        """
        Import raw text directly.

        Args:
            text: Raw text content
            source: Source identifier for metadata

        Returns:
            LoaderResult with Document
        """
        doc = Document(
            page_content=text, metadata={"source": source, "source_type": "text"}
        )

        documents = [doc]
        if self.chunker:
            documents = self.chunker.chunk_batch(documents)

        return LoaderResult(documents=documents)


# Convenience exports
__all__ = [
    "DocumentImporter",
    "DocumentConverter",
    "URLFetcher",
    "TextChunker",
    "LoaderResult",
    "ConversionResult",
]
