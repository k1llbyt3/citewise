
import pytest

import citewise.semantic_retriever as semantic_module
from citewise.embeddings import EMBEDDING_DIMENSIONS
from citewise.semantic_retriever import (
    SemanticRetriever,
    cosine_similarity,
)


def make_vector(first, second=0.0):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = first
    vector[1] = second
    return vector


def make_index():
    return {
        "dimensions": EMBEDDING_DIMENSIONS,
        "items": [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "page_number": 1,
                "filename": "guide.pdf",
                "text": "Unrelated passage.",
                "embedding": make_vector(0.0, 1.0),
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-1",
                "page_number": 2,
                "filename": "guide.pdf",
                "text": "Passage about password security.",
                "embedding": make_vector(1.0, 0.1),
            },
            {
                "chunk_id": "chunk-3",
                "document_id": "doc-2",
                "page_number": 4,
                "filename": "other.pdf",
                "text": "Another unrelated passage.",
                "embedding": make_vector(-1.0, 0.0),
            },
        ],
    }


def test_identical_vectors_have_similarity_one():
    vector = make_vector(1.0, 2.0)

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity(
        make_vector(1.0, 0.0),
        make_vector(0.0, 1.0),
    ) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    assert cosine_similarity(
        make_vector(1.0, 0.0),
        make_vector(-1.0, 0.0),
    ) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError, match="same dimensions"):
        cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite numeric"):
        cosine_similarity([float("nan")], [1.0])


def test_search_ranks_most_similar_chunk_first(monkeypatch):
    monkeypatch.setattr(
        semantic_module,
        "embed_query",
        lambda query, client=None: make_vector(1.0, 0.0),
    )

    retriever = SemanticRetriever(make_index())
    results = retriever.search("password protection", top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "chunk-2"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["filename"] == "guide.pdf"
    assert results[0]["page_number"] == 2
    assert results[0]["text"] == "Passage about password security."


def test_search_respects_top_k(monkeypatch):
    monkeypatch.setattr(
        semantic_module,
        "embed_query",
        lambda query, client=None: make_vector(1.0, 0.0),
    )

    retriever = SemanticRetriever(make_index())

    assert len(retriever.search("passwords", top_k=1)) == 1


def test_blank_query_returns_no_results_without_embedding(monkeypatch):
    def unexpected_embedding(*args, **kwargs):
        raise AssertionError("Blank queries should not call the embedder.")

    monkeypatch.setattr(semantic_module, "embed_query", unexpected_embedding)

    retriever = SemanticRetriever(make_index())

    assert retriever.search("   ") == []


def test_empty_index_returns_no_results_without_embedding(monkeypatch):
    def unexpected_embedding(*args, **kwargs):
        raise AssertionError("Empty indexes should not call the embedder.")

    monkeypatch.setattr(semantic_module, "embed_query", unexpected_embedding)

    retriever = SemanticRetriever({
        "dimensions": EMBEDDING_DIMENSIONS,
        "items": [],
    })

    assert retriever.search("password security") == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2"])
def test_search_rejects_invalid_top_k(top_k, monkeypatch):
    monkeypatch.setattr(
        semantic_module,
        "embed_query",
        lambda query, client=None: make_vector(1.0, 0.0),
    )

    retriever = SemanticRetriever(make_index())

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.search("passwords", top_k=top_k)


def test_search_rejects_non_string_query():
    retriever = SemanticRetriever(make_index())

    with pytest.raises(TypeError, match="Query must be a string"):
        retriever.search(None)


def test_retriever_rejects_wrong_index_dimensions():
    with pytest.raises(ValueError, match="unexpected embedding dimensions"):
        SemanticRetriever({"dimensions": 2, "items": []})


def test_retriever_rejects_missing_metadata():
    index = {
        "dimensions": EMBEDDING_DIMENSIONS,
        "items": [{"text": "Missing source information"}],
    }

    with pytest.raises(ValueError, match="missing required metadata"):
        SemanticRetriever(index)


def test_retriever_rejects_wrong_vector_dimensions():
    item = make_index()["items"][0]
    item["embedding"] = [0.1, 0.2]

    with pytest.raises(ValueError, match="wrong vector dimensions"):
        SemanticRetriever({
            "dimensions": EMBEDDING_DIMENSIONS,
            "items": [item],
        })
