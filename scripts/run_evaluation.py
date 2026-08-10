import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests

from policypilot.services.citation_validator import extract_citation_urls
from policypilot.services.source_policy import is_trusted_source_url


API_URL = os.getenv("POLICYPILOT_BACKEND_URL", "http://localhost:8000/api").rstrip("/")
DATASET = Path("evaluation/questions.json")
REFUSAL_MARKERS = ("couldn't verify", "could not verify", "official eligibility")


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    totals = {"retrieval_hit": 0, "trusted_sources": 0, "valid_citations": 0, "correct_refusal": 0}

    for case in cases:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "thread_id": f"eval-{uuid.uuid4()}",
                "user_profile": {
                    "state": "Texas",
                    "age": 40,
                    "household_size": 2,
                    "income": 50000,
                    "medical_history": None,
                },
                "message": case["question"],
                "conversation_history": [],
                "is_profile_complete": True,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["agent_response"]
        urls = [source["url"] for source in payload.get("sources", [])]
        domains = {urlsplit(url).hostname or "" for url in urls}
        expected_hit = not case["expected_domains"] or any(
            any(domain == expected or domain.endswith(f".{expected}") for expected in case["expected_domains"])
            for domain in domains
        )
        citation_urls = extract_citation_urls(answer)
        refused = any(marker in answer.lower() for marker in REFUSAL_MARKERS)

        totals["retrieval_hit"] += int(expected_hit)
        totals["trusted_sources"] += int(all(is_trusted_source_url(url) for url in urls))
        totals["valid_citations"] += int(
            (case["should_refuse"] and refused)
            or (bool(citation_urls) and citation_urls.issubset(set(urls)))
        )
        totals["correct_refusal"] += int(refused == case["should_refuse"])
        print(f"{case['id']}: sources={len(urls)} refused={refused}")

    count = len(cases)
    print(json.dumps({key: round(value / count, 3) for key, value in totals.items()}, indent=2))


if __name__ == "__main__":
    main()
