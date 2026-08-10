import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from policypilot.api.endpoints import router as api_router
from policypilot.config import settings, validate_runtime_settings
from policypilot.core.agent_orchestrator import build_orchestrator
from policypilot.services.database import database_is_ready, setup_database
from policypilot.services.vector_store import get_vector_store_service


logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PolicyPilot")
    validate_runtime_settings()
    setup_database()
    app.state.vector_store = get_vector_store_service()
    app.state.orchestrator = build_orchestrator()
    yield
    logger.info("Stopping PolicyPilot")


app = FastAPI(
    title="PolicyPilot API",
    description="A source-grounded U.S. health insurance advisor powered by agentic RAG.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Welcome to PolicyPilot. API documentation is available at /docs."}


@app.get("/health")
async def health():
    sqlite_ready = database_is_ready()
    vector_store = getattr(app.state, "vector_store", None)
    chroma_ready = bool(vector_store and vector_store.is_ready())
    status = "ok" if sqlite_ready and chroma_ready else "degraded"
    payload = {
        "status": status,
        "checks": {"application": True, "sqlite": sqlite_ready, "chroma": chroma_ready},
    }
    return JSONResponse(content=payload, status_code=200 if status == "ok" else 503)
