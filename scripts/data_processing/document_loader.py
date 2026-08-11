import logging
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
from typing import Optional

logger = logging.getLogger(__name__)

# Loads and extracts clean text content from an HTML file.
def load_html_content(file_path: Path) -> Optional[str]:
    """Loads and extracts clean text content from an HTML file."""
    logger.debug(f"Loading HTML from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')

        # Remove script, style, nav, footer, header, and other common clutter
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()

        # Get text, strip whitespace, and join lines
        text = ' '.join(soup.get_text(separator=' ', strip=True).split())

        if not text:
            logger.warning(f"No text content could be extracted from {file_path}")
            return None
        return text
    except Exception as e:
        logger.error(f"Failed to load or parse HTML file {file_path}: {e}")
        return None

# Loads and extracts text content from a PDF file.
def load_pdf_content(file_path: Path) -> Optional[str]:
    """Loads and extracts text content from a PDF file."""
    logger.debug(f"Loading PDF from: {file_path}")
    if not file_path.exists():
        logger.error(f"PDF file not found at {file_path}")
        return None
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n" # Add space between pages

        if not text:
            logger.warning(f"No text could be extracted from PDF {file_path}")
            return None
        return text
    except Exception as e:
        logger.error(f"Failed to load or parse PDF file {file_path}: {e}")
        return None

# Generic document loader that dispatches to the correct function.
def load_document(file_path_str: str) -> Optional[str]:
    """
    Generic document loader that dispatches to the correct function
    based on file extension.
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"Document not found at path: {file_path}")
        return None

    extension = file_path.suffix.lower()
    if extension == '.html':
        return load_html_content(file_path)
    elif extension == '.pdf':
        return load_pdf_content(file_path)
    else:
        logger.warning(f"Unsupported file type '{extension}' for file {file_path}. Skipping.")
        return None
