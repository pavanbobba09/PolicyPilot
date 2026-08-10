import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from policypilot.api.endpoints import router
from policypilot.core import agent_orchestrator
from policypilot import main


class FakeOrchestrator:
    def invoke(self, inputs, config):
        return {
            **inputs,
            "generation": "Answer ([CMS](https://www.cms.gov/rules)).",
            "conversation_history": ["Agent: Answer"],
            "sources": [{"name": "CMS", "url": "https://www.cms.gov/rules"}],
        }


def test_chat_contract_sources_and_sensitive_profile_redaction(caplog):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.orchestrator = FakeOrchestrator()
    secret = "private-medication-name"
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "thread_id": "thread-1",
                "user_profile": {"medications": [secret]},
                "message": "Question",
                "conversation_history": [],
                "is_profile_complete": True,
            },
        )
    assert response.status_code == 200
    assert response.json()["sources"] == [{"name": "CMS", "url": "https://www.cms.gov/rules"}]
    assert secret not in caplog.text


def test_invalid_zip_is_rejected_without_external_lookup():
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get("/api/geodata/not-a-zip")
    assert response.status_code == 400


def test_chat_provider_failure_returns_bounded_error():
    class BrokenOrchestrator:
        def invoke(self, inputs, config):
            raise RuntimeError("provider unavailable")

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.orchestrator = BrokenOrchestrator()
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "thread_id": "thread-2",
                "user_profile": {},
                "message": "Question",
                "conversation_history": [],
                "is_profile_complete": True,
            },
        )
    assert response.status_code == 500
    assert "could not complete" in response.json()["detail"]


def test_health_reports_sqlite_and_chroma_readiness(monkeypatch):
    class ReadyVectorStore:
        def is_ready(self):
            return True

    monkeypatch.setattr(main, "database_is_ready", lambda: True)
    main.app.state.vector_store = ReadyVectorStore()
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"application": True, "sqlite": True, "chroma": True},
    }


def test_model_provider_assignments(monkeypatch):
    gemini = object()
    groq = object()
    gemini_pro = object()

    class FakeVector:
        def get_retriever(self):
            return object()

    monkeypatch.setattr(agent_orchestrator, "get_vector_store_service", lambda: FakeVector())
    monkeypatch.setattr(agent_orchestrator.llm_provider, "get_gemini_llm", lambda: gemini)
    monkeypatch.setattr(agent_orchestrator.llm_provider, "get_llama_fast_llm", lambda: groq)
    monkeypatch.setattr(agent_orchestrator.llm_provider, "get_gemini_pro_llm", lambda: gemini_pro)
    monkeypatch.setattr(agent_orchestrator, "ProfileBuilder", lambda llm: ("profile", llm))
    monkeypatch.setattr(agent_orchestrator, "QueryTransformationAgent", lambda llm, retriever: ("transform", llm))
    monkeypatch.setattr(agent_orchestrator, "RouterAgent", lambda llm: ("router", llm))
    monkeypatch.setattr(agent_orchestrator, "SearchAgent", lambda llm: ("search", llm))
    monkeypatch.setattr(agent_orchestrator, "IngestionService", lambda store: ("ingest", store))
    monkeypatch.setattr(agent_orchestrator, "AdvisorAgent", lambda llm: ("advisor", llm))

    deps = agent_orchestrator.build_dependencies()
    assert deps.profile_builder == ("profile", gemini)
    assert deps.transformer == ("transform", groq)
    assert deps.router == ("router", groq)
    assert deps.searcher == ("search", gemini)
    assert deps.advisor == ("advisor", gemini_pro)
