import logging

from langchain_core.documents import Document

from policypilot.config import settings
from policypilot.services.database import get_db_connection, setup_database
from policypilot.services.ingestion_service import IngestionService
from policypilot.services.vector_store import get_vector_store_service


logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Explicitly ingest crawler output; application startup never crawls or ingests.
def main() -> None:
    """Explicitly ingest crawler output; application startup never crawls or ingests."""

    setup_database()
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM data_sources
            WHERE status IN ('processed', 'updated') AND local_path IS NOT NULL
            """
        ).fetchall()

    documents = [
        Document(
            page_content="",
            metadata={
                "source_url": row["url"],
                "source_name": row["name"],
                "source_local_path": row["local_path"],
                "content_hash": row["content_hash"],
                "retrieved_at": row["updated_at"],
            },
        )
        for row in rows
    ]
    result = IngestionService(get_vector_store_service()).ingest_documents(documents)
    logger.info("PolicyPilot ingestion result: %s", result.model_dump())


if __name__ == "__main__":
    main()
