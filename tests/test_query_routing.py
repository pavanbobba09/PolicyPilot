from types import SimpleNamespace

from langchain_core.documents import Document

from policypilot.core.agents.query_transformer import QueryTransformationAgent
from policypilot.core.models import IntentType, TransformedQueries


class FakeClassifier:
    def __init__(self, intent):
        self.intent = intent

    def classify_intent(self, _query):
        return SimpleNamespace(intent=self.intent, reasoning="test")


class FakeChain:
    def __init__(self, queries):
        self.queries = queries

    def invoke(self, _inputs):
        return TransformedQueries(transformed_queries=self.queries)


class FakeRetriever:
    def __init__(self):
        self.invoked = []
        self.batched = []
        self.document = Document(
            page_content="coverage",
            metadata={"source_id": 1, "source_url": "https://www.cms.gov/rules"},
        )

    def invoke(self, query):
        self.invoked.append(query)
        return [self.document]

    def batch(self, queries):
        self.batched.append(queries)
        return [[self.document] for _ in queries]


def make_agent(intent, transformed_queries):
    agent = QueryTransformationAgent.__new__(QueryTransformationAgent)
    agent.classifier = FakeClassifier(intent)
    agent.chain = FakeChain(transformed_queries)
    agent.retriever = FakeRetriever()
    return agent


def test_simple_query_uses_single_retrieval():
    agent = make_agent(IntentType.SIMPLE, ["original"])
    agent.transform_and_retrieve("original")
    assert agent.retriever.invoked == ["original"]
    assert agent.retriever.batched == []


def test_ambiguous_query_uses_rag_fusion_batch():
    agent = make_agent(IntentType.AMBIGUOUS, ["variation one", "variation two"])
    documents = agent.transform_and_retrieve("original")
    assert agent.retriever.batched == [["original", "variation one", "variation two"]]
    assert len(documents) == 1
