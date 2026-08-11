# policypilot/core/agents/query_intent_classifier.py

import logging
import pydantic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from policypilot.core.models import IntentType, QueryIntent
from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUERY_INTENT_PROMPT_TEMPLATE = load_prompt('query_intent_classifier')


class QueryIntentClassifierAgent:
    """
    A classifier that uses an LLM to determine the user's intent and
    pre-processes the query for optimal RAG performance.
    """
    # Initializes the classifier with a given LLM.
    def __init__(self, llm):
        """
        Initializes the classifier with a given LLM.
        Generated code
        Args:
            llm: An instance of a LLM model.
        """
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=QueryIntent)
        self.prompt_template = PromptTemplate(
            template=QUERY_INTENT_PROMPT_TEMPLATE,
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "simple": IntentType.SIMPLE.value,
                "ambiguous": IntentType.AMBIGUOUS.value,
                "complex": IntentType.COMPLEX.value,
                "concise": IntentType.CONCISE.value,
            },
        )
        self.chain = self.prompt_template | self.llm | self.parser
        logger.info("QueryIntentClassifierAgent initialized successfully.")

    # Classifies the intent of the query and transforms it accordingly.
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classifies the intent of the query and transforms it accordingly.

        Args:
            query: The user's input query.

        Returns:
            A QueryIntent object containing the classification and transformed queries.
            Returns a default 'SIMPLE' classification on failure.
        """
        if not query:
            logger.warning("Received an empty query. Cannot classify.")
            return QueryIntent(
                intent=IntentType.SIMPLE,
                reasoning="Input query was empty.",
            )

        logger.debug("Classifying an advisor query")
        try:
            result = self.chain.invoke({"query": query})
            logger.info(f"Successfully classified query. Intent: {result.intent.value}")
            return result
        except (pydantic.ValidationError, Exception) as e:
            logger.error(
                f"Failed to classify or parse the provider response. Error: {e}. "
                "Defaulting to 'SIMPLE' intent."
            )
            # Fallback mechanism to ensure the RAG chain doesn't break
            return QueryIntent(
                intent=IntentType.SIMPLE,
                reasoning="Classification failed due to a parsing or API error. Defaulting to simple retrieval.",
            )
