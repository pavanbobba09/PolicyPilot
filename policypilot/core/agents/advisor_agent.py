import json
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from policypilot.prompts.prompt_loader import load_prompt
from policypilot.services.citation_validator import allowed_source_urls, citations_are_valid


logger = logging.getLogger(__name__)
GROUNDING_FAILURE = (
    "I couldn't verify enough information from PolicyPilot's approved government sources "
    "to answer that safely. Please confirm the details through HealthCare.gov or the "
    "relevant government benefits program."
)


class AdvisorAgent:
    """Generate a personalized answer and enforce source-backed citations."""

    # Initialize this object and its required dependencies.
    def __init__(self, llm):
        """Initialize this object and its required dependencies."""
        self.llm = llm
        self.agent_prompt = load_prompt("advisor_agent")

    # Build the grounded answer prompt from the question, profile, and evidence.
    def _prompt(self, question: str, user_profile: Dict[str, Any], documents: List[Document]) -> str:
        """Build the grounded answer prompt from the question, profile, and evidence."""
        context = "\n\n---\n\n".join(
            f"[METADATA: source_name='{document.metadata.get('source_name', 'N/A')}', "
            f"source_url='{document.metadata.get('source_url', 'N/A')}']\n\n{document.page_content}"
            for document in documents
        )
        return (
            f"{self.agent_prompt}\n\n### CONTEXT FOR YOUR RESPONSE\n"
            f"user_profile: {json.dumps(user_profile, indent=2)}\n\n"
            f"user_question: {json.dumps(question)}\n\n"
            f"retrieved_context:\n{context}"
        )

    # Generate a cited response, retry invalid citations once, and fail safely.
    def generate_response(
        self,
        question: str,
        user_profile: Dict[str, Any],
        documents: List[Document],
    ) -> str:
        """Generate a cited response, retry invalid citations once, and fail safely."""
        if not documents:
            return GROUNDING_FAILURE

        base_prompt = self._prompt(question, user_profile, documents)
        try:
            first = self.llm.invoke(base_prompt).content.strip()
            if citations_are_valid(first, documents):
                return first

            allowed = sorted(allowed_source_urls(documents))
            retry_prompt = (
                f"{base_prompt}\n\n### CITATION CORRECTION\n"
                "The previous response omitted citations or used a URL outside the retrieved context. "
                "Regenerate the answer once. Every factual claim must use Markdown citations, and the "
                f"only permitted citation URLs are: {json.dumps(allowed)}"
            )
            second = self.llm.invoke(retry_prompt).content.strip()
            return second if citations_are_valid(second, documents) else GROUNDING_FAILURE
        except Exception:
            logger.exception("Gemini answer generation failed")
            return "PolicyPilot couldn't reach its answer-generation provider. Please try again shortly."
