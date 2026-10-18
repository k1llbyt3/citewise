"""Unit tests for the query-embedding cache module."""

import json
from pathlib import Path
import pytest

from citewise.embeddings import EMBEDDING_DIMENSIONS, MODEL_NAME
from citewise.query_cache import (
    QueryCacheMissError,
    QueryEmbeddingCache,
    main as query_cache_main,
    populate_cache_from_dataset,
)


def make_vector(val: float = 1.0) -> list[float]:
    vec = [0.0] * EMBEDDING_DIMENSIONS
    vec[0] = val
    return vec


def test_empty_cache_miss_raises_in_offline_mode(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    cache = QueryEmbeddingCache(cache_path=cache_file, allow_api=False)

    with pytest.raises(QueryCacheMissError, match="In offline mode, live API calls are forbidden"):
        cache.get_embedding("What is NIST AI RMF?")


def test_cache_miss_populates_and_saves_when_api_allowed(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    called = []

    def mock_embed(query: str):
        called.append(query)
        return make_vector(0.5)

    cache = QueryEmbeddingCache(
        cache_path=cache_file,
        allow_api=True,
        embed_fn=mock_embed,
    )

    vec1 = cache.get_embedding("Test Query")
    assert called == ["Test Query"]
    assert len(vec1) == EMBEDDING_DIMENSIONS
    assert cache_file.exists()

    # Second call should hit the cache without calling embed_fn
    vec2 = cache.get_embedding("Test Query")
    assert len(called) == 1
    assert vec1 == vec2

    # Verify persistent file content
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["model"] == MODEL_NAME
    assert data["dimensions"] == EMBEDDING_DIMENSIONS
    assert "Test Query" in data["queries"]


def test_cache_reloading_and_sharing(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    cache1 = QueryEmbeddingCache(cache_path=cache_file, allow_api=True, embed_fn=lambda q: make_vector(0.9))
    cache1.get_embedding("Shared Query")

    # Load from another cache instance pointing to the same file
    cache2 = QueryEmbeddingCache(cache_path=cache_file, allow_api=False)
    assert "Shared Query" in cache2.queries
    vec = cache2.get_embedding("Shared Query")
    assert vec == make_vector(0.9)


def test_interrupted_run_persists_prior_queries(tmp_path: Path):
    cache_file = tmp_path / "cache.json"
    call_count = 0

    def mock_embed(query: str):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("API connection interrupted!")
        return make_vector(float(call_count))

    cache = QueryEmbeddingCache(cache_path=cache_file, allow_api=True, embed_fn=mock_embed)

    cache.get_embedding("Query 1")
    cache.get_embedding("Query 2")
    with pytest.raises(RuntimeError, match="API connection interrupted"):
        cache.get_embedding("Query 3")

    # Re-open cache: Query 1 and 2 must have survived on disk
    recovered_cache = QueryEmbeddingCache(cache_path=cache_file, allow_api=False)
    assert "Query 1" in recovered_cache.queries
    assert "Query 2" in recovered_cache.queries
    assert "Query 3" not in recovered_cache.queries


def test_reject_invalid_json(tmp_path: Path):
    cache_file = tmp_path / "broken.json"
    cache_file.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid query cache JSON"):
        QueryEmbeddingCache(cache_path=cache_file)


def test_reject_model_mismatch(tmp_path: Path):
    cache_file = tmp_path / "wrong_model.json"
    cache_file.write_text(
        json.dumps({
            "version": 1,
            "model": "text-embedding-004",
            "dimensions": EMBEDDING_DIMENSIONS,
            "queries": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Query cache model mismatch"):
        QueryEmbeddingCache(cache_path=cache_file)


def test_reject_dimension_mismatch(tmp_path: Path):
    cache_file = tmp_path / "wrong_dims.json"
    cache_file.write_text(
        json.dumps({
            "version": 1,
            "model": MODEL_NAME,
            "dimensions": 512,
            "queries": {},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Query cache dimensions mismatch"):
        QueryEmbeddingCache(cache_path=cache_file)


def test_reject_corrupted_vector_in_cache(tmp_path: Path):
    cache_file = tmp_path / "corrupted_entry.json"
    bad_vec = [1.0] * 10  # wrong dimensions
    cache_file.write_text(
        json.dumps({
            "version": 1,
            "model": MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "queries": {"Q": bad_vec},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has 10 dimensions; expected 768"):
        QueryEmbeddingCache(cache_path=cache_file)


def test_reject_non_numeric_or_nan_in_cache(tmp_path: Path):
    cache_file = tmp_path / "nan_entry.json"
    vec = make_vector(0.0)
    vec[5] = None  # None is non-numeric
    cache_file.write_text(
        json.dumps({
            "version": 1,
            "model": MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "queries": {"Q": vec},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains non-numeric value"):
        QueryEmbeddingCache(cache_path=cache_file)


def test_populate_cache_from_dataset(tmp_path: Path):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps({
            "cases": [
                {"id": "c1", "question": "Question 1"},
                {"id": "c2", "question": "Question 2"},
            ]
        }),
        encoding="utf-8",
    )

    cache_file = tmp_path / "cache.json"
    called = []

    def mock_embed(query: str):
        called.append(query)
        return make_vector(1.0)

    count = populate_cache_from_dataset(
        dataset_path=dataset_file,
        cache_path=cache_file,
        embed_fn=mock_embed,
    )
    assert count == 2
    assert len(called) == 2

    # Second run should populate 0 items
    count2 = populate_cache_from_dataset(
        dataset_path=dataset_file,
        cache_path=cache_file,
        embed_fn=mock_embed,
    )
    assert count2 == 0
    assert len(called) == 2


def test_query_cache_main_cli(tmp_path: Path, capsys):
    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text(
        json.dumps({
            "cases": [
                {"id": "c1", "question": "Question 1"},
            ]
        }),
        encoding="utf-8",
    )
    cache_file = tmp_path / "cache.json"

    # Offline check fails when missing
    ret = query_cache_main(["--dataset", str(dataset_file), "--cache", str(cache_file)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "1 query embeddings are missing" in captured.err
