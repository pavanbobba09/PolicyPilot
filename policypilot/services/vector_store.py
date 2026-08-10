import logging
from functools import lru_cache
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from policypilot.config import settings


logger = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ChromaONNXEmbeddings(Embeddings):
    """LangChain adapter for Chroma's CPU-only all-MiniLM ONNX runtime."""

    def __init__(self):
        self._embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._embedding_function(texts)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


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
        logger.info("Loading ONNX embedding model %s", EMBEDDING_MODEL_NAME)
        return ChromaONNXEmbeddings()

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
