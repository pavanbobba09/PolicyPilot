from urllib.parse import urlsplit, urlunsplit

from policypilot.config import TRUSTED_GOVERNMENT_DOMAINS


# Return a stable HTTPS URL without fragments or a redundant trailing slash.
def canonicalize_url(url: str) -> str:
    """Return a stable HTTPS URL without fragments or a redundant trailing slash."""

    value = (url or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if not scheme or not hostname:
        return ""
    port = f":{parts.port}" if parts.port else ""
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, hostname + port, path, parts.query, ""))


# Check whether a URL uses HTTPS and belongs to an approved domain.
def is_trusted_source_url(url: str) -> bool:
    """Check whether a URL uses HTTPS and belongs to an approved domain."""
    canonical = canonicalize_url(url)
    if not canonical:
        return False
    parts = urlsplit(canonical)
    if parts.scheme != "https":
        return False
    hostname = parts.hostname or ""
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in TRUSTED_GOVERNMENT_DOMAINS)


# Return approved domains in the format required by web search.
def trusted_search_domains() -> list[str]:
    """Return approved domains in the format required by web search."""
    return list(TRUSTED_GOVERNMENT_DOMAINS)
