import numpy as np

from policypilot.services import vector_store as vector_store_module


def test_onnx_embedding_adapter_supports_langchain_interface(monkeypatch):
    class FakeONNXEmbeddingFunction:
        def __call__(self, texts):
            return np.asarray([[float(len(text)), 1.0] for text in texts])

    monkeypatch.setattr(
        vector_store_module,
        "DefaultEmbeddingFunction",
        lambda: FakeONNXEmbeddingFunction(),
    )

    embeddings = vector_store_module.ChromaONNXEmbeddings()

    assert embeddings.embed_documents(["ACA", "Medicare"]) == [[3.0, 1.0], [8.0, 1.0]]
    assert embeddings.embed_query("VA") == [2.0, 1.0]
