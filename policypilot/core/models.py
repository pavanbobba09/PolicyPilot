import logging
from enum import Enum
from typing import List, Optional, Any, Type, Dict
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    """
    Enum for the types of user query intents.
    This provides a controlled vocabulary for our RAG strategies.
    """
    SIMPLE = "Simple"
    AMBIGUOUS = "Ambiguous (RAG-Fusion)"
    COMPLEX = "Complex (Decomposition)"
    CONCISE = "Concise (Step-Back)"

class QueryIntent(BaseModel):
    """
    Pydantic model for the structured output of the query intent classifier.
    This model ensures the LLM's response is predictable and directly usable.
    """
    intent: IntentType = Field(
        ...,
        description="The classified intent of the user's query."
    )
    reasoning: str = Field(
        ...,
        description="A brief explanation of why the query was classified with this intent."
    )
    transformed_queries: List[str] = Field(
        default_factory=list,
        description=(
            "The transformed query/queries suitable for the classified intent. "
            "For SIMPLE, this is the original query. "
            "For AMBIGUOUS, these are multiple variations for RAG-Fusion. "
            "For COMPLEX, these are the decomposed sub-queries. "
            "For CONCISE, this is a single 'step-back' question."
        )
    )

class TransformedQueries(BaseModel):
    """
    Pydantic model for the structured output of the query transformation step.
    """
    transformed_queries: List[str] = Field(
        ...,
        description=(
            "A list of transformed queries suitable for the classified intent. "
            "For SIMPLE, this should be the original query in a list. "
            "For AMBIGUOUS, these should be multiple variations for RAG-Fusion. "
            "For COMPLEX, these should be the decomposed sub-queries. "
            "For SPECULATIVE, this should be a single hypothetical document in a list. "
            "For CONCISE, this should be a single 'step-back' question in a list."
        )
    )

class UserProfile(BaseModel):
    """
    Pydantic model for the user's 360-degree profile.
    """
    # --- Mandatory fields provided initially ---
    zip_code: str = Field(..., description="User's 5-digit ZIP code.")
    county: str = Field(..., description="User's county, derived from ZIP code.")
    state: str = Field(..., description="User's state, derived from ZIP code.")
    age: int = Field(..., description="User's age in years.")
    gender: str = Field(..., description="User's gender (e.g., 'Male', 'Female', 'Other').")
    household_size: int = Field(..., description="Number of people in the user's household.")
    income: int = Field(..., description="Annual household income.")
    employment_status: str = Field(..., description="User's employment status.")
    citizenship: str = Field(..., description="User's citizenship status.")

    # --- Fields to be populated by the ProfileAgent ---
    medical_history: Optional[List[str]] = Field(
        None, description="List of chronic conditions or significant medical history."
    )
    medications: Optional[List[str]] = Field(
        None, description="List of current prescription medications."
    )
    special_cases: Optional[List[str]] = Field(
        None, description="Special circumstances like pregnancy, tobacco use, etc."
    )

class GeoDataResponse(BaseModel):
    """
    The response model for the /geodata endpoint.
    """
    zip_code: str
    county: str
    city: str
    state: str
    state_abbreviation: str

class ChatRequest(BaseModel):
    """
    The request model for the main /chat endpoint.
    """
    thread_id: str
    user_profile: Dict[str, Optional[Any]]
    message: str
    conversation_history: List[str] = Field(default_factory=list)
    is_profile_complete: bool


class SourceReference(BaseModel):
    """A trusted source used to ground an advisor response."""

    name: str
    url: str

class ChatResponse(BaseModel):
    """
    The response model for the /chat endpoint.
    """
    agent_response: str
    updated_profile: Dict[str, Any]
    updated_history: List[str]
    is_profile_complete: bool
    sources: List[SourceReference] = Field(default_factory=list)


class IngestionResult(BaseModel):
    """Chunk-level outcome counts from an ingestion operation."""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
