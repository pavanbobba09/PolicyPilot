import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


TRUSTED_GOVERNMENT_DOMAINS = (
    "healthcare.gov",
    "cms.gov",
    "medicaid.gov",
    "medicare.gov",
    "va.gov",
    "tricare.mil",
)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")

    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    DATABASE_URL: str = "data/policypilot.db"
    CHECKPOINT_DATABASE_URL: str = "data/checkpoints.db"
    CHROMA_PATH: str = "data/vector_store"
    CHROMA_COLLECTION: str = "policypilot_kb"
    LOG_LEVEL: str = "INFO"

    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL_NAME: str = "llama-3.1-8b-instant"
    GEMINI_PRO_MODEL_NAME: str = "gemini-3.6-flash"
    GEMINI_MODEL_NAME: str = "gemini-3.5-flash-lite"
    GEMINI_FAST_MODEL_NAME: str = "gemini-3.5-flash-lite"

    CRAWLING_JOBS: List[dict] = [
        {
            "name": "HealthCare.gov Crawl",
            "start_url": "https://www.healthcare.gov/",
            "method": "requests_crawl",
            "domain_lock": "www.healthcare.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active",
        },
        {
            "name": "CMS.gov Regulations & Guidance Crawl",
            "start_url": "https://www.cms.gov/regulations-and-guidance",
            "method": "selenium_crawl",
            "domain_lock": "www.cms.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active",
        },
        {
            "name": "Medicaid.gov Crawl",
            "start_url": "https://www.medicaid.gov/",
            "method": "requests_crawl",
            "domain_lock": "www.medicaid.gov",
            "crawl_depth": 2,
            "content_types": ["pdf", "html"],
            "status": "active",
        },
        {
            "name": "Medicare.gov Crawl",
            "start_url": "https://www.medicare.gov/",
            "method": "selenium_crawl",
            "domain_lock": "www.medicare.gov",
            "crawl_depth": 1,
            "content_types": ["html"],
            "status": "active",
        },
        {
            "name": "TRICARE Publications Crawl",
            "start_url": "https://www.tricare.mil/publications",
            "method": "selenium_crawl",
            "domain_lock": "www.tricare.mil",
            "crawl_depth": 1,
            "content_types": ["pdf", "html"],
            "status": "active",
        },
        {
            "name": "VA.gov Health Benefits Crawl",
            "start_url": "https://www.va.gov/health-care/",
            "method": "requests_crawl",
            "domain_lock": "www.va.gov",
            "crawl_depth": 2,
            "content_types": ["html"],
            "status": "active",
        },
    ]


def validate_runtime_settings(*, require_search: bool = True) -> None:
    """Fail startup with actionable messages when provider credentials are absent."""

    missing = []
    if not settings.GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY (Gemini profile, reformulation, search, and answer generation)")
    if not settings.GROQ_API_KEY:
        missing.append("GROQ_API_KEY (classification, transformation, grading, and summarization)")
    if require_search and not settings.TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY (trusted web-search fallback)")
    if missing:
        raise RuntimeError("Missing required PolicyPilot configuration: " + "; ".join(missing))


settings = Settings()
