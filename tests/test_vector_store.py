from policypilot.services import vector_store as vector_store_module


def test_gemini_embedding_adapter_uses_retrieval_tasks_and_fixed_dimensions(monkeypatch):
    calls = []

    class FakeGeminiEmbeddingClient:
        def embed_documents(self, texts, **kwargs):
            calls.append(("documents", texts, kwargs))
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text, **kwargs):
            calls.append(("query", text, kwargs))
            return [float(len(text)), 1.0]

    monkeypatch.setattr(vector_store_module.settings, "GEMINI_EMBEDDING_DIMENSIONS", 384)
    embeddings = vector_store_module.GeminiEmbeddings(client=FakeGeminiEmbeddingClient())

    assert embeddings.embed_documents(["ACA", "Medicare"]) == [[3.0, 1.0], [8.0, 1.0]]
    assert embeddings.embed_query("VA") == [2.0, 1.0]
    assert calls == [
        (
            "documents",
            ["ACA", "Medicare"],
            {"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 384},
        ),
        (
            "query",
            "VA",
            {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 384},
        ),
    ]
