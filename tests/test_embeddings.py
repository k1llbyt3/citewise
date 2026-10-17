
from types import SimpleNamespace

import pytest

from citewise.embeddings import (
    DEFAULT_BATCH_SIZE,
    DOCUMENT_TASK,
    EMBEDDING_DIMENSIONS,
    QUERY_TASK,
    embed_document,
    embed_query,
    embed_texts,
)


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

        # Match the interface used by the Gemini SDK.
        self.models = self

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        contents = kwargs["contents"]

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[float(len(text))] * EMBEDDING_DIMENSIONS
                )
                for text in contents
            ]
        )


def test_embed_texts_returns_one_vector_per_input():
    client = FakeEmbeddingClient()

    vectors = embed_texts(
        ["First passage", "Second passage"],
        client=client,
    )

    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)
    assert len(client.calls) == 1


def test_embed_texts_splits_inputs_into_batches():
    client = FakeEmbeddingClient()

    vectors = embed_texts(
        ["one", "two", "three"],
        batch_size=2,
        client=client,
    )

    assert len(vectors) == 3
    assert len(client.calls) == 2
    assert client.calls[0]["contents"] == ["one", "two"]
    assert client.calls[1]["contents"] == ["three"]


def test_embed_texts_returns_empty_list_without_api_call():
    client = FakeEmbeddingClient()

    assert embed_texts([], client=client) == []
    assert client.calls == []


def test_embed_document_uses_document_task():
    client = FakeEmbeddingClient()

    vector = embed_document("A PDF passage", client=client)

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert client.calls[0]["config"].task_type == DOCUMENT_TASK


def test_embed_query_uses_query_task():
    client = FakeEmbeddingClient()

    vector = embed_query("What does the PDF say?", client=client)

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert client.calls[0]["config"].task_type == QUERY_TASK


@pytest.mark.parametrize(
    "texts",
    [
        "a single string",
        ["valid text", ""],
        ["valid text", "   "],
        ["valid text", None],
    ],
)
def test_embed_texts_rejects_invalid_inputs(texts):
    client = FakeEmbeddingClient()

    with pytest.raises((TypeError, ValueError)):
        embed_texts(texts, client=client)

    assert client.calls == []


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_embed_texts_rejects_invalid_batch_size(batch_size):
    client = FakeEmbeddingClient()

    with pytest.raises(ValueError, match="batch_size"):
        embed_texts(
            ["some text"],
            batch_size=batch_size,
            client=client,
        )

    assert client.calls == []


def test_embed_texts_rejects_unknown_task_type():
    client = FakeEmbeddingClient()

    with pytest.raises(ValueError, match="Unsupported task_type"):
        embed_texts(
            ["some text"],
            task_type="UNKNOWN_TASK",
            client=client,
        )

    assert client.calls == []


def test_embed_texts_rejects_wrong_vector_dimensions():
    class WrongDimensionClient:
        def __init__(self):
            self.models = self

        def embed_content(self, **kwargs):
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2])]
            )

    with pytest.raises(RuntimeError, match="Expected 768 dimensions"):
        embed_texts(["some text"], client=WrongDimensionClient())


def test_embed_texts_rejects_missing_embeddings():
    class EmptyResponseClient:
        def __init__(self):
            self.models = self

        def embed_content(self, **kwargs):
            return SimpleNamespace(embeddings=None)

    with pytest.raises(RuntimeError, match="no embeddings"):
        embed_texts(["some text"], client=EmptyResponseClient())
