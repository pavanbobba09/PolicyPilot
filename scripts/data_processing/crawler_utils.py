import hashlib
import re
from urllib.parse import urlparse
from pathlib import Path

def get_content_hash(content: bytes) -> str:
    """Generates a SHA-256 hash for the given binary content."""
    return hashlib.sha256(content).hexdigest()

def sanitize_filename(url: str) -> str:
    """
    Creates a sanitized, readable filename from a URL.
    e.g., 'https://example.com/path/to/file.html?query=1' -> 'path_to_file.html'
    """
    parsed_url = urlparse(url)
    # Use the path, but remove leading/trailing slashes
    path_part = parsed_url.path.strip('/')

    if not path_part:
        # If path is empty (e.g., domain.com/), use the netloc
        path_part = parsed_url.netloc

    # Replace slashes with underscores and remove other invalid chars
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', path_part)

    # Limit length to avoid OS errors
    return sanitized[:150]