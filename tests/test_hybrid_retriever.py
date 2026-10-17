
import pytest

import citewise.hybrid_retriever as hybrid_module
from citewise.hybrid_retriever import HybridRetriever


def test_rrf_combines_rankings(monkeypatch):
    class FakeBM25:
        chunks = [
            {"chunk_id": "a", "text": "A"},
            {"chunk_id": "b", "text": "B"},
        ]

        def __init__(self, chunks):
            self.chunks = chunks

        def search(self, query, top_k=5):
            return [
                {"chunk_id": "a", "text": "A", "score": 2.0},
                {"chunk_id": "b", "text": "B", "score": 1.0},
            ][:top_k]

    class FakeSemantic:
        items = [
            {"chunk_id": "b", "text": "B"},
            {"chunk_id": "c", "text": "C"},
        ]

        def __init__(self, index, client=None):
            self.items = index["items"]

        def search(self, query, top_k=5):
            return [
                {"chunk_id": "b", "text": "B", "score": 0.9},
                {"chunk_id": "c", "text": "C", "score": 0.8},
            ][:top_k]

    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(hybrid_module, "SemanticRetriever", FakeSemantic)

    retriever = HybridRetriever(
        chunks=[{"chunk_id": "a"}, {"chunk_id": "b"}],
        index={"items": [{"chunk_id": "b"}, {"chunk_id": "c"}]},
    )

    results = retriever.search("example", top_k=3)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["bm25_rank"] == 2
    assert results[0]["semantic_rank"] == 1
    assert results[0]["hybrid_score"] == pytest.approx(
        1 / 62 + 1 / 61
    )
    assert len(results) == 3


def test_blank_query_returns_empty_results(monkeypatch):
    class FakeBM25:
        chunks = []

        def __init__(self, chunks):
            self.chunks = chunks

    class FakeSemantic:
        items = []

        def __init__(self, index, client=None):
            self.items = index["items"]

    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(hybrid_module, "SemanticRetriever", FakeSemantic)

    retriever = HybridRetriever([], {"items": []})

    assert retriever.search("   ") == []


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        HybridRetriever.__new__(HybridRetriever).search("example", top_k=0)


def test_invalid_rrf_k_is_rejected():
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever([], {"items": []}, rrf_k=0)
