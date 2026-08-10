from types import SimpleNamespace

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver

from policypilot.core.agent_orchestrator import OrchestratorDependencies, build_orchestrator
from policypilot.core.agents.advisor_agent import GROUNDING_FAILURE
from policypilot.core.models import IngestionResult


def document(label):
    return Document(
        page_content=label,
        metadata={"source_name": "CMS", "source_url": "https://www.cms.gov/rules"},
    )


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content=prompt.split("### Follow-up Question\n")[-1])


class FakeTransformer:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def transform_and_retrieve(self, _question):
        result = self.batches[self.calls]
        self.calls += 1
        return result


class FakeRouter:
    def __init__(self, grades):
        self.grades = list(grades)
        self.calls = 0

    def grade_documents(self, _question, _documents):
        result = self.grades[self.calls]
        self.calls += 1
        return result


class FakeSearcher:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = 0

    def search(self, _question):
        self.calls += 1
        return self.results


class FakeIngestor:
    def __init__(self):
        self.calls = 0

    def ingest_documents(self, documents):
        self.calls += 1
        return IngestionResult(added=len(documents))


class FakeAdvisor:
    def __init__(self):
        self.calls = 0
        self.documents = []

    def generate_response(self, _question, _profile, documents):
        self.calls += 1
        self.documents = documents
        return "Grounded ([CMS](https://www.cms.gov/rules))."


def run_graph(transform_batches, grades, search_results=None):
    transformer = FakeTransformer(transform_batches)
    router = FakeRouter(grades)
    searcher = FakeSearcher(search_results)
    ingestor = FakeIngestor()
    advisor = FakeAdvisor()
    deps = OrchestratorDependencies(
        profile_builder=object(),
        reformulation_llm=FakeLLM(),
        transformer=transformer,
        router=router,
        searcher=searcher,
        ingestor=ingestor,
        advisor=advisor,
    )
    graph = build_orchestrator(deps, checkpointer=MemorySaver())
    state = graph.invoke(
        {
            "user_profile": {"state": "Texas"},
            "user_message": "Am I eligible?",
            "conversation_history": [],
            "is_profile_complete": True,
            "web_search_attempted": False,
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )
    return state, transformer, router, searcher, ingestor, advisor


def test_relevant_documents_generate_without_web_search():
    state, transformer, _, searcher, ingestor, advisor = run_graph([[document("initial")]], [True])
    assert state["generation"].startswith("Grounded")
    assert transformer.calls == 1
    assert searcher.calls == 0
    assert ingestor.calls == 0
    assert advisor.calls == 1


def test_web_fallback_ingests_and_retrieves_fresh_documents():
    fresh = document("fresh")
    state, transformer, router, searcher, ingestor, advisor = run_graph(
        [[document("stale")], [fresh]],
        [False, True],
        [fresh],
    )
    assert state["is_relevant"] is True
    assert transformer.calls == router.calls == 2
    assert searcher.calls == ingestor.calls == 1
    assert advisor.documents == [fresh]


def test_web_fallback_is_bounded_and_refuses_insufficient_evidence():
    state, transformer, router, searcher, ingestor, advisor = run_graph(
        [[document("stale")], [document("still stale")]],
        [False, False],
        [],
    )
    assert state["generation"] == GROUNDING_FAILURE
    assert transformer.calls == router.calls == 2
    assert searcher.calls == ingestor.calls == 1
    assert advisor.calls == 0
