from langchain_core.documents import Document

from policypilot.config import settings
from policypilot.services.database import setup_database
from policypilot.services.ingestion_service import IngestionService


class FakeVectorStore:
    def __init__(self):
        self.documents = {}
        self.deleted = []

    def add_documents(self, documents, ids=None):
        for vector_id, document in zip(ids, documents):
            self.documents[vector_id] = document
        return ids

    def delete(self, ids):
        self.deleted.extend(ids)
        for vector_id in ids:
            self.documents.pop(vector_id, None)


def search_document(path):
    return Document(
        page_content="",
        metadata={
            "source_url": "https://www.cms.gov/coverage",
            "source_name": "CMS Coverage",
            "source_local_path": str(path),
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        },
    )


def test_ingestion_uses_deterministic_ids_and_skips_unchanged_content(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", str(tmp_path / "policypilot.db"))
    setup_database()
    source_file = tmp_path / "coverage.html"
    source_file.write_text("<html><body>Coverage rule one.</body></html>", encoding="utf-8")
    vector_store = FakeVectorStore()
    service = IngestionService(vector_store)

    first = service.ingest_documents([search_document(source_file)])
    first_ids = set(vector_store.documents)
    second = service.ingest_documents([search_document(source_file)])

    assert first.added > 0
    assert second.skipped == first.added
    assert set(vector_store.documents) == first_ids

    source_file.write_text("<html><body>Coverage rule two changed.</body></html>", encoding="utf-8")
    third = service.ingest_documents([search_document(source_file)])
    assert third.updated > 0
    assert set(vector_store.documents) != first_ids
    assert first_ids.issubset(set(vector_store.deleted))
