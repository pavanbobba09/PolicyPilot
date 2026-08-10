import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document

from policypilot.core.models import IngestionResult
from policypilot.services.database import (
    find_or_create_web_source,
    get_source_by_url,
    get_vector_ids_for_source,
    replace_source_chunks,
)
from policypilot.services.source_policy import canonicalize_url, is_trusted_source_url
from scripts.data_processing.chunker import chunk_text
from scripts.data_processing.document_loader import load_document


logger = logging.getLogger(__name__)


class IngestionService:
    """Load, deduplicate, chunk, and persist trusted web-search documents."""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _vector_id(url: str, content_hash: str, index: int) -> str:
        source_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"source_{source_key}_{content_hash[:16]}_{index:04d}"

    def ingest_documents(self, documents_from_search: List[Document]) -> IngestionResult:
        counts = {"added": 0, "updated": 0, "skipped": 0, "failed": 0}

        for search_document in documents_from_search:
            metadata = search_document.metadata
            source_url = canonicalize_url(str(metadata.get("source_url", "")))
            source_name = str(metadata.get("source_name") or "Government source")
            local_path = str(metadata.get("source_local_path") or "")

            if not is_trusted_source_url(source_url) or not local_path:
                logger.warning("Skipping untrusted or incomplete search result: %s", source_url)
                counts["failed"] += 1
                continue

            content = load_document(local_path)
            if not content:
                counts["failed"] += 1
                continue

            content_hash = self._content_hash(content)
            existing = get_source_by_url(source_url)
            source_id = find_or_create_web_source(source_url, source_name)
            old_vector_ids = get_vector_ids_for_source(source_id)

            if existing and existing.get("content_hash") == content_hash and old_vector_ids:
                counts["skipped"] += len(old_vector_ids)
                continue

            retrieved_at = str(
                metadata.get("retrieved_at")
                or datetime.now(timezone.utc).isoformat()
            )
            source_metadata = {
                "id": source_id,
                "url": source_url,
                "name": source_name,
                "local_path": local_path,
                "content_hash": content_hash,
                "retrieved_at": retrieved_at,
            }
            chunks = chunk_text(content, source_metadata)
            if not chunks:
                counts["failed"] += 1
                continue

            vector_ids = [
                self._vector_id(source_url, content_hash, index)
                for index in range(len(chunks))
            ]
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.metadata["chunk_id"] = vector_id
                chunk.metadata["content_hash"] = content_hash
                chunk.metadata["retrieved_at"] = retrieved_at

            try:
                # A previous interrupted attempt may already have these deterministic IDs.
                self.vector_store.delete(vector_ids)
                self.vector_store.add_documents(chunks, ids=vector_ids)
                stale_ids = [vector_id for vector_id in old_vector_ids if vector_id not in vector_ids]
                self.vector_store.delete(stale_ids)
                replace_source_chunks(
                    source_id,
                    name=source_name,
                    local_path=local_path,
                    content_hash=content_hash,
                    chunks=(
                        (chunk.page_content, json.dumps(chunk.metadata), vector_id)
                        for chunk, vector_id in zip(chunks, vector_ids)
                    ),
                )
            except Exception:
                logger.exception("Failed to ingest %s", source_url)
                counts["failed"] += len(chunks)
                continue

            key = "updated" if existing and existing.get("content_hash") else "added"
            counts[key] += len(chunks)

        return IngestionResult(**counts)
