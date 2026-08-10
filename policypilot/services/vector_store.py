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

    def __init__(self, client: Optional[GoogleGenerativeAIEmbeddings] = None):
        self._client = client or GoogleGenerativeAIEmbeddings(
            model=settings.GEMINI_EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        self._dimensions = settings.GEMINI_EMBEDDING_DIMENSIONS

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=self._dimensions,
        )

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._dimensions,
        )


class VectorStoreService:
    def __init__(self, embedding_function: Optional[Embeddings] = None):
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

    def _get_embedding_function(self) -> Embeddings:
        logger.info(
            "Using hosted Gemini embedding model %s (%s dimensions)",
            settings.GEMINI_EMBEDDING_MODEL_NAME,
            settings.GEMINI_EMBEDDING_DIMENSIONS,
        )
        return GeminiEmbeddings()

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        if not documents:
            return []
        return self.langchain_chroma.add_documents(documents=documents, ids=ids)

    def delete(self, ids: List[str]) -> None:
        if ids:
            self.langchain_chroma.delete(ids=ids)

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        return self.langchain_chroma.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs or {"k": 5},
        )

    def is_ready(self) -> bool:
        try:
            self.client.get_or_create_collection(self.collection_name).count()
            return True
        except Exception:
            logger.exception("ChromaDB readiness check failed")
            return False


@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService()
