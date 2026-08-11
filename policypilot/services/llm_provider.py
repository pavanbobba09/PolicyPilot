import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from policypilot.config import settings


logger = logging.getLogger(__name__)


# Create a configured Gemini chat model for the requested role.
def _gemini(model: str) -> ChatGoogleGenerativeAI:
    """Create a configured Gemini chat model for the requested role."""
    logger.info("Initializing Gemini model %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )


# Create a configured Groq chat model for the requested role.
def _groq(model: str) -> ChatGroq:
    """Create a configured Groq chat model for the requested role."""
    logger.info("Initializing Groq model %s", model)
    return ChatGroq(
        model=model,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )


# Return the higher-capability Gemini model used for final synthesis.
def get_gemini_pro_llm() -> ChatGoogleGenerativeAI:
    """Return the higher-capability Gemini model used for final synthesis."""
    return _gemini(settings.GEMINI_PRO_MODEL_NAME)


# Return the default Gemini model used by general agent tasks.
def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Return the default Gemini model used by general agent tasks."""
    return _gemini(settings.GEMINI_MODEL_NAME)


# Return the low-latency Gemini model used for lightweight tasks.
def get_gemini_fast_llm() -> ChatGoogleGenerativeAI:
    """Return the low-latency Gemini model used for lightweight tasks."""
    return _gemini(settings.GEMINI_FAST_MODEL_NAME)


# Return the primary Groq-hosted Llama model.
def get_llama_llm() -> ChatGroq:
    """Return the primary Groq-hosted Llama model."""
    return _groq(settings.GROQ_MODEL_NAME)


# Return the low-latency Groq-hosted Llama model.
def get_llama_fast_llm() -> ChatGroq:
    """Return the low-latency Groq-hosted Llama model."""
    return _groq(settings.GROQ_FAST_MODEL_NAME)
