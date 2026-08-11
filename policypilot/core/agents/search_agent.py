import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit

import requests
from langchain_core.documents import Document
from tavily import TavilyClient

from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt
from policypilot.services.source_policy import (
    canonicalize_url,
    is_trusted_source_url,
    trusted_search_domains,
)


logger = logging.getLogger(__name__)
DYNAMIC_DATA_DIR = Path("data/dynamic")


# Convert arbitrary source text into a filesystem-safe filename.
def sanitize_filename(name: str) -> str:
    """Convert arbitrary source text into a filesystem-safe filename."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)[:150]


# Infer a supported document extension from a source URL.
def get_file_extension_from_url(url: str) -> str:
    """Infer a supported document extension from a source URL."""
    return ".pdf" if urlsplit(url).path.lower().endswith(".pdf") else ".html"


class SearchAgent:
    """Search only approved government domains and save results for ingestion."""

    # Initialize this object and its required dependencies.
    def __init__(self, llm, tavily_client: Optional[TavilyClient] = None):
        """Initialize this object and its required dependencies."""
        self.llm = llm
        self.query_prompt = load_prompt("search_agent")
        self.tavily_client = tavily_client or TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PolicyPilot/1.0 (+local portfolio project)"})
        DYNAMIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Turn a user question into a focused health-insurance search query.
    def _formulate_query(self, user_question: str) -> Optional[str]:
        """Turn a user question into a focused health-insurance search query."""
        try:
            response = self.llm.invoke(f"{self.query_prompt}\n\nUser Question: {user_question}")
            query = response.content.strip()
            return None if query == "NOT_RELEVANT" else query
        except Exception:
            logger.exception("Web-search query formulation failed")
            return None

    # Persist a downloaded search result locally for subsequent ingestion.
    def _save_result_to_file(self, result: dict) -> Optional[Path]:
        """Persist a downloaded search result locally for subsequent ingestion."""
        url = canonicalize_url(str(result.get("url", "")))
        if not is_trusted_source_url(url):
            logger.warning("Rejected untrusted search result: %s", url)
            return None

        extension = get_file_extension_from_url(url)
        url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        save_path = DYNAMIC_DATA_DIR / f"{sanitize_filename(url_key)}{extension}"
        try:
            if extension == ".pdf":
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                save_path.write_bytes(response.content)
            else:
                content = str(result.get("raw_content") or result.get("content") or "")
                if not content:
                    return None
                save_path.write_text(content, encoding="utf-8")
            return save_path
        except (requests.RequestException, OSError):
            logger.exception("Failed to save trusted search result %s", url)
            return None

    # Search trusted government domains and return locally saved documents.
    def search(self, user_question: str) -> List[Document]:
        """Search trusted government domains and return locally saved documents."""
        search_query = self._formulate_query(user_question)
        if not search_query:
            return []
        try:
            response = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                include_raw_content=True,
                include_domains=trusted_search_domains(),
                max_results=5,
            )
        except Exception:
            logger.exception("Tavily search failed")
            return []

        documents: List[Document] = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        for result in response.get("results", []):
            url = canonicalize_url(str(result.get("url", "")))
            if not is_trusted_source_url(url):
                continue
            local_path = self._save_result_to_file(result)
            if not local_path:
                continue
            content = str(result.get("raw_content") or result.get("content") or "")
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source_url": url,
                        "source_name": str(result.get("title") or "Government source"),
                        "source_local_path": str(local_path),
                        "retrieved_at": retrieved_at,
                        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    },
                )
            )
        return documents
