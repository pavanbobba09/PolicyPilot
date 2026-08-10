import re
from typing import Iterable

from langchain_core.documents import Document

from policypilot.core.models import SourceReference
from policypilot.services.source_policy import canonicalize_url, is_trusted_source_url


MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")


def extract_citation_urls(text: str) -> set[str]:
    return {canonicalize_url(url) for url in MARKDOWN_LINK_PATTERN.findall(text or "")}


def allowed_source_urls(documents: Iterable[Document]) -> set[str]:
    return {
        canonicalize_url(str(document.metadata.get("source_url", "")))
        for document in documents
        if is_trusted_source_url(str(document.metadata.get("source_url", "")))
    }


def citations_are_valid(text: str, documents: Iterable[Document]) -> bool:
    allowed = allowed_source_urls(documents)
    cited = extract_citation_urls(text)
    return bool(allowed and cited) and cited.issubset(allowed)


def source_references(documents: Iterable[Document]) -> list[SourceReference]:
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
