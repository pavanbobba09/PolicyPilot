import logging
from typing import List
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Pydantic model for the structured output from the LLM
class GradeDocuments(BaseModel):
    """Binary score for document relevance."""
    is_relevant: str = Field(description="Is the document relevant to the question? Answer 'yes' or 'no'.")

class RouterAgent:
    """
    An agent that acts as a router by grading the relevance of retrieved documents.
    It decides whether the existing knowledge is sufficient to answer a question.
    """

    def __init__(self, llm):
        """Initializes the RouterAgent."""
        try:
            self.grader_prompt = load_prompt("document_grader")
            # Create a structured LLM instance that is constrained to the GradeDocuments schema
            self.structured_llm_grader = llm.with_structured_output(GradeDocuments)
            logger.info("RouterAgent (Document Grader) initialized successfully.")
        except FileNotFoundError:
            logger.critical("Document grader prompt file not found. The RouterAgent cannot function.")
            raise
        except Exception as e:
            logger.critical(f"Failed to initialize RouterAgent: {e}")
            raise

    def grade_documents(self, question: str, documents: List[Document]) -> bool:
        """
        Grades the retrieved documents against the user's question.

        Args:
            question: The user's question.
            documents: A list of documents retrieved from the vector store.

        Returns:
            True if the documents are relevant, False otherwise.
        """
        if not documents:
            logger.warning("No documents provided to grade. Assuming they are not relevant.")
            return False

        # Combine the content of all documents into a single context string
        document_text = "\n\n---\n\n".join([d.page_content for d in documents])

        full_prompt = f"{self.grader_prompt}\n\n[USER QUESTION]\n{question}\n\n[DOCUMENTS]\n{document_text}"

        logger.debug("Grading document relevance...")
        try:
            grade = self.structured_llm_grader.invoke(full_prompt)
            is_relevant = grade.is_relevant == "yes"

            if is_relevant:
                logger.info("GRADE: Documents are RELEVANT.")
            else:
                logger.warning("GRADE: Documents are NOT RELEVANT.")

            return is_relevant
        except Exception as e:
            logger.error(f"Error during document grading: {e}. Defaulting to 'not relevant'.")
            # Fail-safe: if grading fails, it's safer to assume the docs are not relevant
            # and trigger a web search to get fresh information.
            return False
