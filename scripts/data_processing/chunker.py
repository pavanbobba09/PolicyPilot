import logging
import json
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def chunk_text(text: str, source_metadata: Dict[str, Any]) -> List[Document]:
    """
    Chunks the given text and attaches rich metadata to each chunk.

    Args:
        text: The full text content to be chunked.
        source_metadata: A dictionary containing metadata about the source document
                         (e.g., id, url, local_path).

    Returns:
        A list of LangChain Document objects, each representing a chunk.
    """
    if not text:
        logger.warning(f"Received empty text for source_id {source_metadata.get('id')}. No chunks created.")
        return []

    # Using RecursiveCharacterTextSplitter as it's robust for general text.
    # These parameters can be tuned based on embedding model's context window and performance.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_texts = text_splitter.split_text(text)

    documents = []
    for i, chunk_text in enumerate(split_texts):
        # This metadata is crucial for the Fairness Agent and for filtering.
        chunk_metadata = {
            "source_id": source_metadata.get("id"),
            "source_url": source_metadata.get("url"),
            "source_name": source_metadata.get("name"),
            "source_local_path": source_metadata.get("local_path"),
            "chunk_number": i + 1,
            "total_chunks": len(split_texts),
            "retrieved_at": source_metadata.get("retrieved_at") or source_metadata.get("updated_at"),
            "content_hash": source_metadata.get("content_hash"),
            "chunk_id": source_metadata.get("chunk_id"),
        }

        doc = Document(page_content=chunk_text, metadata=chunk_metadata)
        documents.append(doc)

    logger.info(f"Created {len(documents)} chunks for source_id {source_metadata.get('id')}")
    return documents
