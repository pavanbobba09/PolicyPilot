from urllib.parse import urlsplit, urlunsplit

from policypilot.config import TRUSTED_GOVERNMENT_DOMAINS


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


def is_trusted_source_url(url: str) -> bool:
    canonical = canonicalize_url(url)
    if not canonical:
        return False
    parts = urlsplit(canonical)
    if parts.scheme != "https":
        return False
    hostname = parts.hostname or ""
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in TRUSTED_GOVERNMENT_DOMAINS)


def trusted_search_domains() -> list[str]:
    return list(TRUSTED_GOVERNMENT_DOMAINS)
