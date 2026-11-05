"""Tests for document loader integrations."""

import subprocess
from pathlib import Path

import pytest
from langchain_core.documents import Document

from vector_graph_rag.loaders import (
    ConversionResult,
    DoclingConverter,
    DocumentImporter,
    MinerUConverter,
    URLFetcher,
)
from vector_graph_rag.loaders import docling as docling_module
from vector_graph_rag.loaders import mineru as mineru_module
from vector_graph_rag.loaders import url_fetcher as url_fetcher_module


class FakeConverter:
    """Simple converter used to test importer injection."""

    supported_extensions = {".pdf", ".pptx"}

    def __init__(self, content: str = "Parsed content"):
        self.content = content
        self.sources: list[str] = []

    def convert(self, source: str) -> ConversionResult:
        self.sources.append(source)
        return ConversionResult(
            documents=[
                Document(
                    page_content=self.content,
                    metadata={"source": source, "parser": "fake"},
                )
            ]
        )

    def convert_batch(self, sources: list[str]) -> ConversionResult:
        documents = []
        errors = []
        for source in sources:
            result = self.convert(source)
            documents.extend(result.documents)
            errors.extend(result.errors)
        return ConversionResult(documents=documents, errors=errors)


class FakeDoclingDocument:
    """Simple Docling document stand-in for converter tests."""

    def __init__(self, markdown: str):
        self.markdown = markdown
        self.export_calls: list[dict] = []

    def export_to_markdown(self, **kwargs):
        self.export_calls.append(kwargs)
        return self.markdown


class FakeDoclingResult:
    """Simple Docling conversion result stand-in."""

    def __init__(self, document):
        self.document = document


class FakeDoclingConverter:
    """Simple Docling converter stand-in."""

    def __init__(self, document=None, error: Exception | None = None):
        self.document = document
        self.error = error
        self.sources: list[str] = []

    def convert(self, source: str):
        self.sources.append(source)
        if self.error:
            raise self.error
        return FakeDoclingResult(self.document)


def test_document_importer_uses_injected_converter(tmp_path: Path):
    source = tmp_path / "slides.pptx"
    source.write_bytes(b"not a real presentation")
    converter = FakeConverter()

    importer = DocumentImporter(chunk_documents=False, converter=converter)
    result = importer.import_sources([str(source)])

    assert result.errors == []
    assert converter.sources == [str(source)]
    assert len(result.documents) == 1
    assert result.documents[0].page_content == "Parsed content"
    assert result.documents[0].metadata["source"] == str(source)
    assert result.documents[0].metadata["parser"] == "fake"


def test_document_importer_chunks_converted_documents_and_preserves_metadata(tmp_path: Path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"not a real pdf")
    converter = FakeConverter(content="alpha beta gamma delta epsilon zeta")

    importer = DocumentImporter(
        chunk_documents=True,
        chunk_size=12,
        chunk_overlap=0,
        converter=converter,
    )
    result = importer.import_sources([str(source)])

    assert result.errors == []
    assert len(result.documents) > 1
    assert {doc.metadata["source"] for doc in result.documents} == {str(source)}
    assert {doc.metadata["parser"] for doc in result.documents} == {"fake"}
    assert [doc.metadata["chunk_index"] for doc in result.documents] == list(
        range(len(result.documents))
    )


def test_document_importer_rejects_extensions_not_supported_by_converter(tmp_path: Path):
    source = tmp_path / "sheet.xlsx"
    source.write_bytes(b"not a real spreadsheet")

    importer = DocumentImporter(chunk_documents=False, converter=FakeConverter())
    result = importer.import_sources([str(source)])

    assert result.documents == []
    assert result.errors == ["Unsupported file type: .xlsx"]


def test_url_fetcher_reuses_injected_converter_for_pdf_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    converter = FakeConverter(content="PDF from URL")

    class FakeResponse:
        content = b"pdf bytes"

        def raise_for_status(self) -> None:
            return None

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(url_fetcher_module.requests, "get", fake_get)

    fetcher = URLFetcher(converter=converter)
    result = fetcher.fetch("https://example.com/manual.pdf")

    assert result.errors == []
    assert len(result.documents) == 1
    assert result.documents[0].page_content == "PDF from URL"
    assert result.documents[0].metadata["source"] == "https://example.com/manual.pdf"
    assert result.documents[0].metadata["source_type"] == "pdf_url"
    assert len(converter.sources) == 1


def test_docling_converter_requires_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(docling_module, "HAS_DOCLING", False)
    monkeypatch.setattr(docling_module, "DoclingDocumentConverter", None)

    with pytest.raises(ImportError, match="docling"):
        DoclingConverter()


def test_docling_converter_converts_markdown_output(tmp_path: Path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")
    fake_document = FakeDoclingDocument("# Report\n\nParsed body")
    fake_converter = FakeDoclingConverter(document=fake_document)

    converter = DoclingConverter(
        converter=fake_converter,
        export_kwargs={"image_placeholder": "<image>"},
    )
    result = converter.convert(str(source))

    assert result.errors == []
    assert len(result.documents) == 1
    assert result.documents[0].page_content == "# Report\n\nParsed body"
    assert result.documents[0].metadata == {
        "source": str(source),
        "source_type": "pdf",
        "parser": "docling",
    }
    assert fake_converter.sources == [str(source)]
    assert fake_document.export_calls == [{"image_placeholder": "<image>"}]


def test_docling_converter_returns_error_when_no_document_is_found(tmp_path: Path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")
    converter = DoclingConverter(converter=FakeDoclingConverter(document=None))

    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"Docling completed but no document output was found for {source}"]


def test_docling_converter_returns_error_when_output_is_empty(tmp_path: Path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")
    converter = DoclingConverter(converter=FakeDoclingConverter(document=FakeDoclingDocument(" ")))

    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"Docling produced an empty Markdown document for {source}"]


def test_docling_converter_handles_missing_and_unsupported_files(tmp_path: Path):
    converter = DoclingConverter(converter=FakeDoclingConverter())

    missing_result = converter.convert(str(tmp_path / "missing.pdf"))
    assert missing_result.documents == []
    assert missing_result.errors == [f"File not found: {tmp_path / 'missing.pdf'}"]

    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello", encoding="utf-8")
    unsupported_result = converter.convert(str(unsupported))
    assert unsupported_result.documents == []
    assert unsupported_result.errors == ["Unsupported file type for Docling: .txt"]


def test_docling_converter_convert_batch_aggregates_results(tmp_path: Path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    missing = tmp_path / "missing.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class BatchDoclingConverter:
        def convert(self, source: str):
            source_path = Path(source)
            return FakeDoclingResult(FakeDoclingDocument(f"# {source_path.stem}"))

    converter = DoclingConverter(converter=BatchDoclingConverter())
    result = converter.convert_batch([str(first), str(missing), str(second)])

    assert [doc.page_content for doc in result.documents] == ["# first", "# second"]
    assert result.errors == [f"File not found: {missing}"]


def test_docling_converter_returns_error_when_conversion_fails(tmp_path: Path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")
    converter = DoclingConverter(converter=FakeDoclingConverter(error=RuntimeError("parse failed")))

    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"Failed to convert {source} with Docling: parse failed"]


def test_mineru_converter_requires_cli(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: None)

    with pytest.raises(ImportError, match="MinerU CLI"):
        MinerUConverter(command="missing-mineru")


def test_mineru_converter_converts_markdown_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")
    calls = []

    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: "/usr/bin/mineru")

    def fake_run(self, command):
        calls.append(
            {
                "command": command,
                "timeout": self.timeout,
            }
        )
        output_root = Path(command[command.index("-o") + 1])
        nested_output = output_root / source.stem / "auto"
        nested_output.mkdir(parents=True)
        (nested_output / "other.md").write_text("wrong output", encoding="utf-8")
        (nested_output / f"{source.stem}.md").write_text(
            "# Report\n\nParsed body", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(MinerUConverter, "_run_command", fake_run)

    converter = MinerUConverter(
        output_dir=str(tmp_path / "mineru-output"),
        timeout=15,
        extra_args=["--lang", "en"],
        check_command=False,
    )
    result = converter.convert(str(source))

    assert result.errors == []
    assert len(result.documents) == 1
    assert result.documents[0].page_content == "# Report\n\nParsed body"
    assert result.documents[0].metadata == {
        "source": str(source),
        "source_type": "pdf",
        "parser": "mineru",
    }
    assert calls == [
        {
            "command": [
                "/usr/bin/mineru",
                "-p",
                str(source),
                "-o",
                str(tmp_path / "mineru-output"),
                "--lang",
                "en",
            ],
            "timeout": 15,
        }
    ]


def test_mineru_converter_returns_error_when_cli_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")

    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: "/usr/bin/mineru")
    monkeypatch.setattr(
        MinerUConverter,
        "_run_command",
        lambda self, command: subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="parse failed",
        ),
    )

    converter = MinerUConverter(check_command=False)
    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"MinerU failed to convert {source}: parse failed"]


def test_mineru_converter_returns_error_when_no_markdown_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")

    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: "/usr/bin/mineru")
    monkeypatch.setattr(
        MinerUConverter,
        "_run_command",
        lambda self, command: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    converter = MinerUConverter(check_command=False)
    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"MinerU completed but no Markdown output was found for {source}"]


def test_mineru_converter_handles_missing_and_unsupported_files(tmp_path: Path):
    converter = MinerUConverter(check_command=False)

    missing_result = converter.convert(str(tmp_path / "missing.pdf"))
    assert missing_result.documents == []
    assert missing_result.errors == [f"File not found: {tmp_path / 'missing.pdf'}"]

    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello", encoding="utf-8")
    unsupported_result = converter.convert(str(unsupported))
    assert unsupported_result.documents == []
    assert unsupported_result.errors == ["Unsupported file type for MinerU: .txt"]


def test_mineru_converter_convert_batch_aggregates_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    missing = tmp_path / "missing.pdf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: "/usr/bin/mineru")

    def fake_run(self, command):
        source_path = Path(command[command.index("-p") + 1])
        output_root = Path(command[command.index("-o") + 1])
        output_dir = output_root / source_path.stem
        output_dir.mkdir(parents=True)
        (output_dir / f"{source_path.stem}.md").write_text(
            f"# {source_path.stem}",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(MinerUConverter, "_run_command", fake_run)

    converter = MinerUConverter(check_command=False)
    result = converter.convert_batch([str(first), str(missing), str(second)])

    assert [doc.page_content for doc in result.documents] == ["# first", "# second"]
    assert result.errors == [f"File not found: {missing}"]


def test_mineru_converter_returns_error_when_cli_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"pdf bytes")

    monkeypatch.setattr(mineru_module.shutil, "which", lambda command: "/usr/bin/mineru")

    def fake_run(self, command):
        raise subprocess.TimeoutExpired(command, timeout=self.timeout)

    monkeypatch.setattr(MinerUConverter, "_run_command", fake_run)

    converter = MinerUConverter(timeout=1, check_command=False)
    result = converter.convert(str(source))

    assert result.documents == []
    assert result.errors == [f"MinerU timed out converting {source}"]


def test_text_chunker_rejects_overlap_not_smaller_than_chunk_size():
    """Overlap >= chunk_size used to raise inside range() or silently drop text."""
    from vector_graph_rag.loaders.chunker import TextChunker

    for overlap in (100, 200):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=overlap)


def test_text_chunker_rejects_non_positive_chunk_size():
    """chunk_size=0 produced a zero step and raised ValueError from range()."""
    from vector_graph_rag.loaders.chunker import TextChunker

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be greater than or equal to 0"):
        TextChunker(chunk_size=100, chunk_overlap=-1)


def test_text_chunker_character_fallback_keeps_long_unseparated_text():
    """A document without separators must still be chunked, not dropped."""
    from vector_graph_rag.loaders.chunker import TextChunker

    text = "x" * 300
    chunks = TextChunker(chunk_size=100, chunk_overlap=20)._split_text(text)

    assert len(chunks) == 4
    assert chunks[0] == text[:100]
    assert chunks[-1].endswith(text[-20:])
