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


def test_hybrid_retriever_passes_query_cache(monkeypatch):
    received_cache = []

    class FakeSemantic:
        def __init__(self, index, client=None, query_cache=None):
            received_cache.append(query_cache)
            self.items = []

        def search(self, query, top_k=5):
            return []

    class FakeBM25:
        def __init__(self, chunks):
            self.chunks = []

        def search(self, query, top_k=5):
            return []

    monkeypatch.setattr(hybrid_module, "SemanticRetriever", FakeSemantic)
    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25)

    sentinel_cache = object()
    retriever = HybridRetriever([], {"items": []}, query_cache=sentinel_cache)
    assert retriever.query_cache is sentinel_cache
    assert received_cache == [sentinel_cache]


@pytest.mark.parametrize("top_k", [3, 5, 8])
def test_hybrid_retriever_supported_top_k_options(monkeypatch, top_k):
    """Verify Hybrid retriever executes cleanly for all UI-supported top_k values."""
    chunks = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(10)]
    items = [{"chunk_id": f"c{i}", "text": f"text {i}"} for i in range(10)]

    class FakeBM25:
        def __init__(self, c):
            self.chunks = c
        def search(self, query, top_k=5):
            return [{"chunk_id": f"c{i}", "text": f"text {i}", "score": 10 - i} for i in range(len(self.chunks))][:top_k]

    class FakeSemantic:
        def __init__(self, index, client=None, query_cache=None):
            self.items = index["items"]
        def search(self, query, top_k=5):
            return [{"chunk_id": f"c{i}", "text": f"text {i}", "score": 1.0 - i * 0.05} for i in range(len(self.items))][:top_k]

    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(hybrid_module, "SemanticRetriever", FakeSemantic)

    retriever = HybridRetriever(chunks=chunks, index={"items": items})
    results = retriever.search("query", top_k=top_k)
    assert len(results) == top_k
    assert results[0]["chunk_id"] == "c0"


def test_hybrid_retriever_empty_semantic_items_no_crash(monkeypatch):
    """When a document is uploaded without semantic embeddings, Hybrid search must not crash with top_k=0."""
    chunks = [
        {"chunk_id": "c1", "text": "Variables store data.", "score": 5.0},
        {"chunk_id": "c2", "text": "Functions execute logic.", "score": 3.0},
    ]

    class FakeBM25:
        def __init__(self, c):
            self.chunks = c
        def search(self, query, top_k=5):
            return [
                {"chunk_id": "c1", "text": "Variables store data.", "score": 5.0},
                {"chunk_id": "c2", "text": "Functions execute logic.", "score": 3.0},
            ][:top_k]

    class RealSemanticValidator:
        def __init__(self, index, client=None, query_cache=None):
            self.items = index.get("items", [])
        def search(self, query, top_k=5):
            if top_k < 1:
                raise ValueError("top_k must be a positive integer.")
            return []

    monkeypatch.setattr(hybrid_module, "BM25Retriever", FakeBM25)
    monkeypatch.setattr(hybrid_module, "SemanticRetriever", RealSemanticValidator)

    # Empty semantic index items, as happens for newly uploaded PDF
    retriever = HybridRetriever(chunks=chunks, index={"items": []})
    results = retriever.search("what are variables?", top_k=3)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["bm25_rank"] == 1
    assert results[0]["semantic_rank"] is None


def test_hybrid_retriever_empty_bm25_chunks_no_crash(monkeypatch):
    """When chunks is empty, Hybrid search must not crash calling BM25 with top_k=0."""
    class RealBM25Validator:
        def __init__(self, c):
            self.chunks = c
        def search(self, query, top_k=5):
            if top_k < 1:
                raise ValueError("top_k must be a positive integer.")
            return []

    class FakeSemantic:
        def __init__(self, index, client=None, query_cache=None):
            self.items = index["items"]
        def search(self, query, top_k=5):
            return [{"chunk_id": "s1", "text": "semantic text", "score": 0.9}][:top_k]

    monkeypatch.setattr(hybrid_module, "BM25Retriever", RealBM25Validator)
    monkeypatch.setattr(hybrid_module, "SemanticRetriever", FakeSemantic)

    retriever = HybridRetriever(chunks=[], index={"items": [{"chunk_id": "s1"}]})
    results = retriever.search("query", top_k=3)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "s1"


