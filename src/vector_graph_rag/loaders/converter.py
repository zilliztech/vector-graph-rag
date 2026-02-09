"""
Document converter using Microsoft MarkItDown.
Supports text documents only: PDF, DOCX
"""
from typing import List
from pathlib import Path
from pydantic import BaseModel
from langchain_core.documents import Document

try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


class ConversionResult(BaseModel):
    """Result of document conversion."""

    documents: List[Document]
    errors: List[str] = []

    class Config:
        arbitrary_types_allowed = True


class DocumentConverter:
    """
    Convert text documents to Markdown using Microsoft MarkItDown.

    Supported formats:
    - PDF (tables and formatting preserved)
    - DOCX (full document structure)
    """

    def __init__(self):
        if not HAS_MARKITDOWN:
            raise ImportError(
                "markitdown is not installed. "
                "Install with: uv add 'markitdown[pdf,docx]' --optional loaders"
            )

        self.md = MarkItDown(enable_plugins=False)

    def convert(self, source: str) -> ConversionResult:
        """
        Convert a single file to Markdown.

        Args:
            source: File path to convert

        Returns:
            ConversionResult with Document(s) and any errors
        """
        path = Path(source)
        if not path.exists():
            return ConversionResult(
                documents=[], errors=[f"File not found: {source}"]
            )

        try:
            result = self.md.convert(str(path))

            doc = Document(
                page_content=result.text_content,
                metadata={
                    "source": str(path),
                    "source_type": self._get_source_type(path),
                    "title": getattr(result, "title", None),
                },
            )

            return ConversionResult(documents=[doc])

        except Exception as e:
            return ConversionResult(
                documents=[], errors=[f"Failed to convert {source}: {str(e)}"]
            )

    def convert_batch(self, sources: List[str]) -> ConversionResult:
        """Convert multiple files."""
        all_documents = []
        all_errors = []

        for source in sources:
            result = self.convert(source)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        return ConversionResult(documents=all_documents, errors=all_errors)

    def _get_source_type(self, path: Path) -> str:
        """Detect source type from file extension."""
        ext = path.suffix.lower()
        if ext == ".pdf":
            return "pdf"
        elif ext in (".docx", ".doc"):
            return "docx"
        else:
            return "unknown"
