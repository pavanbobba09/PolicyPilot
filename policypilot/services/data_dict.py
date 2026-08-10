from typing import Dict, Iterable

from langchain_core.documents import Document


def build_data_doc_dict(docs: Iterable[Document], summarizer) -> Dict[str, Dict[str, object]]:
    """Attach Groq-generated source summaries without performing work at import time."""

    result: Dict[str, Dict[str, object]] = {}
    for document in docs:
        source_id = str(document.metadata.get("source_id"))
        local_path = str(document.metadata.get("source_local_path") or "")
        result[source_id] = {
            "doc": document,
            "summary": summarizer.get_summary(source_id, local_path),
        }
    return result
