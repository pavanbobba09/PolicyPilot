import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from policypilot.config import settings


logger = logging.getLogger(__name__)


def _gemini(model: str) -> ChatGoogleGenerativeAI:
    logger.info("Initializing Gemini model %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.GOOGLE_API_KEY,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )


def _groq(model: str) -> ChatGroq:
    logger.info("Initializing Groq model %s", model)
    return ChatGroq(
        model=model,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )


def get_gemini_pro_llm() -> ChatGoogleGenerativeAI:
    return _gemini(settings.GEMINI_PRO_MODEL_NAME)


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    return _gemini(settings.GEMINI_MODEL_NAME)


def get_gemini_fast_llm() -> ChatGoogleGenerativeAI:
    return _gemini(settings.GEMINI_FAST_MODEL_NAME)


def get_llama_llm() -> ChatGroq:
    return _groq(settings.GROQ_MODEL_NAME)


def get_llama_fast_llm() -> ChatGroq:
    return _groq(settings.GROQ_FAST_MODEL_NAME)
