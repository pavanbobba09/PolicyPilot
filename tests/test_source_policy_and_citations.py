from types import SimpleNamespace

from langchain_core.documents import Document

from policypilot.core.agents.advisor_agent import AdvisorAgent, GROUNDING_FAILURE
from policypilot.services.citation_validator import citations_are_valid, source_references
from policypilot.services.source_policy import canonicalize_url, is_trusted_source_url


def government_document(url: str = "https://www.cms.gov/rules") -> Document:
    return Document(
        page_content="Marketplace plans follow federal coverage rules.",
        metadata={"source_name": "CMS", "source_url": url},
    )


def test_trusted_domain_policy_rejects_spoofing_and_http():
    assert is_trusted_source_url("https://www.cms.gov/rules")
    assert is_trusted_source_url("https://subdomain.healthcare.gov/help")
    assert not is_trusted_source_url("https://cms.gov.example.com/rules")
    assert not is_trusted_source_url("http://www.cms.gov/rules")
    assert canonicalize_url("https://WWW.CMS.GOV/rules/#details") == "https://www.cms.gov/rules"


def test_citations_must_be_present_and_come_from_retrieved_metadata():
    documents = [government_document()]
    assert citations_are_valid("See [CMS](https://www.cms.gov/rules).", documents)
    assert not citations_are_valid("CMS says this is true.", documents)
    assert not citations_are_valid("See [Blog](https://example.com/post).", documents)
    assert source_references(documents)[0].name == "CMS"


class SequencedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return SimpleNamespace(content=next(self.responses))


def test_advisor_retries_invalid_citations_once():
    llm = SequencedLLM(
        [
            "Unsupported [citation](https://example.com).",
            "Grounded answer ([CMS](https://www.cms.gov/rules)).",
        ]
    )
    answer = AdvisorAgent(llm).generate_response("Question?", {}, [government_document()])
    assert "Grounded answer" in answer
    assert llm.calls == 2


def test_advisor_refuses_after_second_invalid_citation_attempt():
    llm = SequencedLLM(["No citation.", "Still no citation."])
    answer = AdvisorAgent(llm).generate_response("Question?", {}, [government_document()])
    assert answer == GROUNDING_FAILURE
    assert llm.calls == 2
