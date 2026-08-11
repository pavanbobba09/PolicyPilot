import logging
from functools import lru_cache
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from policypilot.config import settings


logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    """LangChain adapter with retrieval-specific Gemini embedding tasks."""

    # Initialize this object and its required dependencies.
    def __init__(self, client: Optional[GoogleGenerativeAIEmbeddings] = None):
        """Initialize this object and its required dependencies."""
        self._client = client or GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        self._dimensions = settings.GEMINI_EMBEDDING_DIMENSIONS

    # Embed stored chunks using the retrieval-document task type.
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed stored chunks using the retrieval-document task type."""
        return self._client.embed_documents(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._dimensions,
        )

    # Embed a user query using the retrieval-query task type.
    def embed_query(self, text: str) -> List[float]:
        """Embed a user query using the retrieval-query task type."""
        return self._client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._dimensions,
        )


class VectorStoreService:
    # Initialize this object and its required dependencies.
    def __init__(self, embedding_function: Optional[Embeddings] = None):
        """Initialize this object and its required dependencies."""
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.embedding_function = embedding_function or self._get_embedding_function()
        self.collection_name = settings.CHROMA_COLLECTION
        self.langchain_chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        logger.info(
            "ChromaDB initialized: collection=%s path=%s",
            self.collection_name,
            settings.CHROMA_PATH,
        )

    # Create the configured hosted Gemini embedding adapter.
    def _get_embedding_function(self) -> Embeddings:
        """Create the configured hosted Gemini embedding adapter."""
        logger.info(
            "Using hosted Gemini embedding model %s (%s dimensions)",
            settings.GEMINI_EMBEDDING_MODEL_NAME,
            settings.GEMINI_EMBEDDING_DIMENSIONS,
        )
        return GeminiEmbeddings()

    # Embed and insert document chunks into ChromaDB.
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Embed and insert document chunks into ChromaDB."""
        if not documents:
            return []
        return self.langchain_chroma.add_documents(documents=documents, ids=ids)

    # Delete vector-store entries matching the supplied identifiers.
    def delete(self, ids: List[str]) -> None:
        """Delete vector-store entries matching the supplied identifiers."""
        if ids:
            self.langchain_chroma.delete(ids=ids)

    # Create an MMR retriever for selecting diverse relevant chunks.
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Create an MMR retriever for selecting diverse relevant chunks."""
        return self.langchain_chroma.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs or {"k": 5},
        )

    # Check whether the configured backing service is accessible.
    def is_ready(self) -> bool:
        """Check whether the configured backing service is accessible."""
        try:
            self.client.get_or_create_collection(self.collection_name).count()
            return True
        except Exception:
            logger.exception("ChromaDB readiness check failed")
            return False


# Return the process-wide cached vector-store service.
@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    """Return the process-wide cached vector-store service."""
    return VectorStoreService()
