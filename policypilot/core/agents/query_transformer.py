# policypilot/core/query_agent.py

import logging
from typing import Any, List, Dict
from collections import defaultdict
from langchain.load import dumps, loads
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from policypilot.core.agents.query_intent_classifier import QueryIntentClassifierAgent
from policypilot.core.models import IntentType, TransformedQueries

from policypilot.config import settings
from policypilot.prompts.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUERY_TRANSFORMER_PROMPT_TEMPLATE = load_prompt('query_transformer')

class QueryTransformationAgent:
    """
    An agent responsible for analyzing a user's query, applying an advanced
    RAG strategy based on the query's intent, and retrieving enhanced context
    from a vector store.
    """

    def __init__(self, llm, retriever):
        """
        Initializes the QueryTransformationAgent.

        Args:
            llm: An instance of the language model to be used
                 for query analysis and transformation.
            retriever: A LangChain retriever runnable connected to the vector store.
        """
        if not llm or not retriever:
            raise ValueError("LLM and retriever must be provided.")

        self.llm = llm
        self.retriever = retriever

        # 1. Create the parser for the transformed queries
        self.parser = PydanticOutputParser(pydantic_object=TransformedQueries)
        # 2. Create the prompt template, injecting the format instructions
        self.prompt = PromptTemplate(
            template=QUERY_TRANSFORMER_PROMPT_TEMPLATE,
            input_variables=["query", "intent", "reasoning"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        # 3. Build the final chain with the parser at the end
        self.chain = self.prompt | self.llm | self.parser

        self.classifier = QueryIntentClassifierAgent(llm)
        logger.info("QueryTransformationAgent initialized successfully.")

    def _unique_union(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Aggregates and merges lists of documents, grouping them by 'source_id'.
        """
        # Step 1: Group all retrieved chunks by their source_id.
        # defaultdict simplifies the grouping logic.
        docs_by_source = defaultdict(list)
        all_docs = [doc for doc_list in doc_lists for doc in doc_list]

        for doc in all_docs:
            source_id = doc.metadata.get('source_id')
            if source_id is not None:
                docs_by_source[source_id].append(doc)
            else:
                logger.warning("Skipping a retrieved document with no source_id")

        logger.debug(f"Grouped {len(all_docs)} chunks into {len(docs_by_source)} unique sources.")

        # Step 2: Process each group to create a single, merged document.
        CHUNK_SEPARATOR = "\n\n--- chunk ---\n\n"
        final_merged_docs: List[Document] = []
        for source_id, chunks in docs_by_source.items():

            # Step 2a: Sort the chunks by their chunk_number to ensure logical order.
            # We use a default of infinity for any chunk missing a number, pushing it to the end.
            sorted_chunks = sorted(chunks, key=lambda d: d.metadata.get('chunk_number', float('inf')))

            # Step 2b: Use the metadata from the first chunk as the base for our new metadata.
            # This is now deterministic because of the sort.
            base_metadata = sorted_chunks[0].metadata.copy()

            # Step 2c: Concatenate the page content from the sorted chunks.
            merged_content = CHUNK_SEPARATOR.join([chunk.page_content for chunk in sorted_chunks])

            # Step 2d: Create a clean, new metadata object for the merged document.
            final_metadata: Dict[str, Any] = {
                # Preserve essential source-level information
                "source_id": base_metadata.get("source_id"),
                "source_url": base_metadata.get("source_url"),
                "source_name": base_metadata.get("source_name"),
                "source_local_path": base_metadata.get("source_local_path"),
                "retrieved_at": base_metadata.get("retrieved_at"),
                "content_hash": base_metadata.get("content_hash"),
                "chunk_ids": [c.metadata.get("chunk_id") for c in sorted_chunks],
                # Add new summary fields
                "merged_chunks_count": len(sorted_chunks),
                "original_chunk_numbers": [c.metadata.get('chunk_number') for c in sorted_chunks]
            }

            # Step 2e: Create the final Document object.
            merged_doc = Document(
                page_content=merged_content,
                metadata=final_metadata
            )
            final_merged_docs.append(merged_doc)

        logger.info(f"Aggregated and merged chunks into {len(final_merged_docs)} final documents.")
        return final_merged_docs

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            # Iterate through each document in the list, with its rank
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Update the score using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            [loads(doc), score]
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results

    def _perform_rag_fusion(self, original_query: str, generated_queries: List[str]) -> List[Document]:
        """Executes the RAG-Fusion strategy."""
        logger.debug("Performing RAG-Fusion with %d generated queries", len(generated_queries))
        all_queries = [original_query] + generated_queries
        retrieval_results = self.retriever.batch(all_queries)
        # return self.reciprocal_rank_fusion(retrieval_results)[0]
        return self._unique_union(retrieval_results)


    def _perform_decomposition(self, sub_queries: List[str]) -> List[Document]:
        """Executes the Decomposition strategy."""
        logger.debug("Performing decomposition with %d sub-queries", len(sub_queries))
        retrieval_results = self.retriever.batch(sub_queries)
        return self._unique_union(retrieval_results)

    def _perform_step_back(self, original_query: str, step_back_query: str) -> List[Document]:
        """Executes the Step-Back strategy."""
        logger.debug("Performing step-back retrieval")
        queries_to_run = [original_query, step_back_query]
        retrieval_results = self.retriever.batch(queries_to_run)
        return self._unique_union(retrieval_results)

    def transform_and_retrieve(self, query: str) -> List[Document]:
        """
        The main method of the agent. It classifies the query, applies the
        appropriate retrieval strategy, and returns the final list of documents.
        """
        logger.info("Starting query transformation and retrieval")

        try:
            # 1. Classify the query to determine the strategy
            classification = self.classifier.classify_intent(query)
            intent = classification.intent
            reasoning = classification.reasoning

            logger.info("Query classified with intent: %s", intent.value)

            result = self.chain.invoke({"query": query, "intent": intent, "reasoning": reasoning})
            transformed_queries = result.transformed_queries
            logger.debug("Generated %d transformed queries", len(transformed_queries))

            documents: List[Document] = []

            # 2. Route to the appropriate retrieval strategy
            if intent == IntentType.AMBIGUOUS:
                documents = self._perform_rag_fusion(query, transformed_queries)

            elif intent == IntentType.COMPLEX:
                documents = self._perform_decomposition(transformed_queries)

            elif intent == IntentType.CONCISE:
                # Step-back provides one transformed query
                step_back_q = transformed_queries[0] if transformed_queries else ""
                documents = self._perform_step_back(query, step_back_q)

            else: # Default to SIMPLE retrieval
                logger.debug("Performing simple retrieval.")
                documents = self.retriever.invoke(query)

            logger.info("Retrieved %d documents", len(documents))
            return documents

        except Exception as e:
            logger.error("Query transformation failed: %s", e, exc_info=True)
            # Fallback to simple retrieval on any catastrophic failure
            try:
                logger.warning("Falling back to simple retrieval due to an error.")
                return self.retriever.invoke(query)
            except Exception as fallback_e:
                logger.critical(f"Fallback retrieval also failed: {fallback_e}")
                return [] # Return empty list if everything fails
