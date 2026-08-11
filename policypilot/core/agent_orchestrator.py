import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from policypilot.config import settings
from policypilot.core.agents.advisor_agent import AdvisorAgent, GROUNDING_FAILURE
from policypilot.core.agents.profile_agent import ProfileBuilder
from policypilot.core.agents.query_transformer import QueryTransformationAgent
from policypilot.core.agents.router_agent import RouterAgent
from policypilot.core.agents.search_agent import SearchAgent
from policypilot.core.models import SourceReference
from policypilot.prompts.prompt_loader import load_prompt
from policypilot.services import llm_provider
from policypilot.services.citation_validator import source_references
from policypilot.services.ingestion_service import IngestionService
from policypilot.services.vector_store import get_vector_store_service


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    user_profile: Dict[str, Any]
    user_message: str
    conversation_history: List[str]
    is_profile_complete: bool
    standalone_question: str
    documents: List[Document]
    is_relevant: bool
    web_search_attempted: bool
    ingestion_result: Dict[str, int]
    generation: str
    sources: List[SourceReference]


@dataclass
class OrchestratorDependencies:
    profile_builder: Any
    reformulation_llm: Any
    transformer: Any
    router: Any
    searcher: Any
    ingestor: Any
    advisor: Any


# Assign explicit jobs to Gemini and Groq providers.
def build_dependencies() -> OrchestratorDependencies:
    """Assign explicit jobs to Gemini and Groq providers."""

    vector_store = get_vector_store_service()
    gemini_flash = llm_provider.get_gemini_llm()
    groq_fast = llm_provider.get_llama_fast_llm()
    return OrchestratorDependencies(
        profile_builder=ProfileBuilder(gemini_flash),
        reformulation_llm=gemini_flash,
        transformer=QueryTransformationAgent(groq_fast, vector_store.get_retriever()),
        router=RouterAgent(groq_fast),
        searcher=SearchAgent(gemini_flash),
        ingestor=IngestionService(vector_store),
        advisor=AdvisorAgent(llm_provider.get_gemini_pro_llm()),
    )


# Create the SQLite checkpointer used to persist graph conversation state.
def _default_checkpointer() -> SqliteSaver:
    """Create the SQLite checkpointer used to persist graph conversation state."""
    checkpoint_path = Path(settings.CHECKPOINT_DATABASE_URL)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return SqliteSaver(connection)


# Build and compile the agent graph for profiling and grounded question answering.
def build_orchestrator(
    dependencies: OrchestratorDependencies | None = None,
    checkpointer=None,
):
    """Build and compile the agent graph for profiling and grounded question answering."""
    deps = dependencies or build_dependencies()

    # Collect or update the user's insurance profile before question answering.
    def profile_builder_node(state: AgentState) -> Dict[str, Any]:
        """Collect or update the user's insurance profile before question answering."""
        profile = state.get("user_profile", {})
        message = state.get("user_message", "")
        history = state.get("conversation_history", [])
        if message == "START_PROFILE_BUILDING":
            response = deps.profile_builder.get_next_question(profile, [])
            return {
                "conversation_history": [f"Agent: {response}"],
                "generation": response,
                "user_profile": profile,
                "is_profile_complete": False,
                "sources": [],
            }

        last_question = history[-1].removeprefix("Agent: ") if history else ""
        updated_profile = deps.profile_builder.update_profile_with_answer(
            profile, last_question, message
        )
        response = deps.profile_builder.get_next_question(
            updated_profile, history + [f"User: {message}"]
        )
        new_history = history + [f"User: {message}", f"Agent: {response}"]
        complete = response == "PROFILE_COMPLETE"
        if complete:
            response = "Great! Your profile is complete. How can I help with your health insurance questions?"
            new_history[-1] = f"Agent: {response}"
        return {
            "user_profile": updated_profile,
            "is_profile_complete": complete,
            "conversation_history": new_history,
            "generation": response,
            "sources": [],
        }

    # Convert the latest message and conversation context into a standalone question.
    def reformulate_query_node(state: AgentState) -> Dict[str, Any]:
        """Convert the latest message and conversation context into a standalone question."""
        profile = state.get("user_profile", {})
        profile_summary = (
            f"State={profile.get('state')}, Age={profile.get('age')}, "
            f"History={profile.get('medical_history')}"
        )
        prompt = load_prompt("query_reformulator")
        full_prompt = (
            f"{prompt}\n\n### User Profile Summary\n{profile_summary}\n\n"
            f"### Conversation History\n{chr(10).join(state.get('conversation_history', []))}\n\n"
            f"### Follow-up Question\n{state.get('user_message', '')}"
        )
        response = deps.reformulation_llm.invoke(full_prompt)
        return {"standalone_question": response.content.strip()}

    # Retrieve relevant knowledge and grade whether it answers the question.
    def retrieve_and_grade_node(state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant knowledge and grade whether it answers the question."""
        question = state.get("standalone_question", "")
        documents = deps.transformer.transform_and_retrieve(question)
        return {
            "documents": documents,
            "is_relevant": deps.router.grade_documents(question, documents),
        }

    # Search approved sources and ingest newly discovered documents.
    def search_and_ingest_node(state: AgentState) -> Dict[str, Any]:
        """Search approved sources and ingest newly discovered documents."""
        documents = deps.searcher.search(state.get("standalone_question", ""))
        result = deps.ingestor.ingest_documents(documents)
        result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        return {"web_search_attempted": True, "ingestion_result": result_dict}

    # Generate a cited answer from retrieved evidence and update chat history.
    def generate_answer_node(state: AgentState) -> Dict[str, Any]:
        """Generate a cited answer from retrieved evidence and update chat history."""
        documents = state.get("documents", [])
        generation = deps.advisor.generate_response(
            state.get("standalone_question", ""),
            state.get("user_profile", {}),
            documents,
        )
        history = state.get("conversation_history", []) + [
            f"User: {state.get('user_message', '')}",
            f"Agent: {generation}",
        ]
        return {
            "generation": generation,
            "conversation_history": history,
            "sources": source_references(documents),
        }

    # Return a grounded refusal when approved evidence remains insufficient.
    def insufficient_evidence_node(state: AgentState) -> Dict[str, Any]:
        """Return a grounded refusal when approved evidence remains insufficient."""
        history = state.get("conversation_history", []) + [
            f"User: {state.get('user_message', '')}",
            f"Agent: {GROUNDING_FAILURE}",
        ]
        return {"generation": GROUNDING_FAILURE, "conversation_history": history, "sources": []}

    # Route to answering, web search, or an insufficient-evidence response.
    def decide_after_grading(state: AgentState) -> str:
        """Route to answering, web search, or an insufficient-evidence response."""
        if state.get("is_relevant"):
            return "generate"
        if not state.get("web_search_attempted", False):
            return "search"
        return "insufficient"

    # Route incomplete profiles to profiling and complete profiles to question answering.
    def decide_entry_point(state: AgentState) -> str:
        """Route incomplete profiles to profiling and complete profiles to question answering."""
        return "qna" if state.get("is_profile_complete") else "profile"

    builder = StateGraph(AgentState)
    builder.add_node("profile_builder", profile_builder_node)
    builder.add_node("reformulate_query", reformulate_query_node)
    builder.add_node("retrieve_and_grade", retrieve_and_grade_node)
    builder.add_node("search_and_ingest", search_and_ingest_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("insufficient_evidence", insufficient_evidence_node)
    builder.set_conditional_entry_point(
        decide_entry_point,
        {"profile": "profile_builder", "qna": "reformulate_query"},
    )
    builder.add_edge("profile_builder", END)
    builder.add_edge("reformulate_query", "retrieve_and_grade")
    builder.add_conditional_edges(
        "retrieve_and_grade",
        decide_after_grading,
        {
            "search": "search_and_ingest",
            "generate": "generate_answer",
            "insufficient": "insufficient_evidence",
        },
    )
    builder.add_edge("search_and_ingest", "retrieve_and_grade")
    builder.add_edge("generate_answer", END)
    builder.add_edge("insufficient_evidence", END)
    return builder.compile(checkpointer=checkpointer or _default_checkpointer())
