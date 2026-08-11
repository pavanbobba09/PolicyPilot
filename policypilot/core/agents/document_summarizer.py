import logging
from typing import Dict, Optional
from langchain_core.prompts import PromptTemplate

from scripts.data_processing.document_loader import load_document

from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUMMARY_PROMPT_TEMPLATE = load_prompt('document_summarizer')

class DocumentSummarizerAgent:
    """
    A service to generate and cache summaries for full source documents.
    """
    # Initialize this object and its required dependencies.
    def __init__(self, llm):
        """Initialize this object and its required dependencies."""
        self.llm = llm
        self.prompt = PromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm
        self._summary_cache: Dict[str, str] = {}
        logger.info("DocumentSummarizer service initialized.")

    # Retrieves or generates a summary for a given source document.
    def get_summary(self, source_id: int, local_path: str) -> Optional[str]:
        """
        Retrieves or generates a summary for a given source document.
        Uses an in-memory cache to avoid re-summarizing the same document.

        Args:
            source_id: The unique ID of the source document.
            local_path: The local file path to the full document content.

        Returns:
            The summary string, or None if summarization fails.
        """
        cache_key = str(source_id)
        if cache_key in self._summary_cache:
            logger.debug(f"Returning cached summary for source_id: {source_id}")
            return self._summary_cache[cache_key]

        logger.info(f"Generating new summary for source_id: {source_id} from path: {local_path}")

        full_content = load_document(local_path)
        if not full_content:
            logger.error(f"Could not load document content from {local_path} to generate summary.")
            return None

        try:
            # Truncate content to fit model context window if necessary
            # A safe number for many models is around 16k tokens, let's use chars as a proxy
            max_chars = 50000
            truncated_content = full_content[:max_chars]

            response = self.chain.invoke({"document_content": truncated_content})
            summary = response.content if hasattr(response, 'content') else str(response)

            self._summary_cache[cache_key] = summary
            logger.info(f"Successfully generated and cached summary for source_id: {source_id}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary for source_id {source_id}: {e}")
            return None


# Create the source summarizer with its designated Groq/Llama model.
def build_document_summarizer() -> DocumentSummarizerAgent:
    """Create the source summarizer with its designated Groq/Llama model."""

    from policypilot.services import llm_provider

    return DocumentSummarizerAgent(llm_provider.get_llama_llm())
