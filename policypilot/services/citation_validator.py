import re
from typing import Iterable

from langchain_core.documents import Document

from policypilot.core.models import SourceReference
from policypilot.services.source_policy import canonicalize_url, is_trusted_source_url


MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")


# Extract HTTP citation targets from Markdown links in an answer.
def extract_citation_urls(text: str) -> set[str]:
    """Extract HTTP citation targets from Markdown links in an answer."""
    return {canonicalize_url(url) for url in MARKDOWN_LINK_PATTERN.findall(text or "")}


# Collect citation URLs permitted by retrieved document metadata.
def allowed_source_urls(documents: Iterable[Document]) -> set[str]:
    """Collect citation URLs permitted by retrieved document metadata."""
    return {
        canonicalize_url(str(document.metadata.get("source_url", "")))
        for document in documents
        if is_trusted_source_url(str(document.metadata.get("source_url", "")))
    }


# Verify that an answer cites only URLs found in its retrieved evidence.
def citations_are_valid(text: str, documents: Iterable[Document]) -> bool:
    """Verify that an answer cites only URLs found in its retrieved evidence."""
    allowed = allowed_source_urls(documents)
    cited = extract_citation_urls(text)
    return bool(allowed and cited) and cited.issubset(allowed)


# Create a deduplicated API source list from retrieved documents.
def source_references(documents: Iterable[Document]) -> list[SourceReference]:
    """Create a deduplicated API source list from retrieved documents."""
    sources: dict[str, SourceReference] = {}
    for document in documents:
        url = canonicalize_url(str(document.metadata.get("source_url", "")))
        if not is_trusted_source_url(url):
            continue
        sources.setdefault(
            url,
            SourceReference(
                name=str(document.metadata.get("source_name") or url),
                url=url,
            ),
        )
    return list(sources.values())
