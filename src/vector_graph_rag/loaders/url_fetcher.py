"""
URL content fetcher using trafilatura.
Direct markdown extraction with full structure preservation.
"""
import trafilatura
import tempfile
import requests
from pathlib import Path
from .converter import ConversionResult, DocumentConverter
from langchain_core.documents import Document


class URLFetcher:
    """
    Fetch and convert web pages to Markdown.

    Uses trafilatura's direct markdown extraction for best results:
    - Preserves links, headers, lists
    - Removes boilerplate (ads, navigation, footers)
    - Main content detection
    """

    # Browser-like User-Agent to avoid anti-scraping blocks
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def __init__(
        self,
        timeout: float = 30.0,
        include_links: bool = True,
        include_images: bool = False,  # Set to False for text-only focus
    ):
        self.timeout = timeout
        self.include_links = include_links
        self.include_images = include_images
        self.converter = DocumentConverter()  # For PDF URLs
        self.headers = {
            'User-Agent': self.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        # Check by URL extension
        if url.lower().endswith('.pdf'):
            return True

        # Check by Content-Type header
        try:
            response = requests.head(url, headers=self.headers, timeout=5, allow_redirects=True)
            content_type = response.headers.get('Content-Type', '').lower()
            return 'application/pdf' in content_type
        except:
            return False

    def _fetch_pdf_url(self, url: str) -> ConversionResult:
        """Download and convert a PDF URL."""
        try:
            # Download PDF to temporary file
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            # Convert PDF using DocumentConverter
            result = self.converter.convert(tmp_path)

            # Update metadata to reflect URL source
            for doc in result.documents:
                doc.metadata['source'] = url
                doc.metadata['source_type'] = 'pdf_url'

            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except:
                pass

            return result

        except Exception as e:
            return ConversionResult(
                documents=[], errors=[f"Failed to fetch PDF from {url}: {str(e)}"]
            )

    def fetch(self, url: str) -> ConversionResult:
        """
        Fetch and convert a URL to Markdown.

        Args:
            url: Web page URL or PDF URL

        Returns:
            ConversionResult with Document
        """
        try:
            # Check if URL points to PDF
            if self._is_pdf_url(url):
                return self._fetch_pdf_url(url)

            # Fetch HTML content for web pages with custom headers
            response = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            html_content = response.text

            if not html_content:
                return ConversionResult(
                    documents=[], errors=[f"Failed to fetch URL: {url}"]
                )

            # Extract content as markdown (best option for text-focused RAG)
            content = trafilatura.extract(
                html_content,
                output_format="markdown",
                include_links=self.include_links,
                include_images=self.include_images,
            )

            if not content:
                return ConversionResult(
                    documents=[], errors=[f"No content extracted from: {url}"]
                )

            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "source_type": "url",
                },
            )

            return ConversionResult(documents=[doc])

        except Exception as e:
            return ConversionResult(
                documents=[], errors=[f"Failed to fetch {url}: {str(e)}"]
            )

    def fetch_batch(self, urls: list[str]) -> ConversionResult:
        """Fetch multiple URLs."""
        all_documents = []
        all_errors = []

        for url in urls:
            result = self.fetch(url)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)

        return ConversionResult(documents=all_documents, errors=all_errors)
