"""Offline end-to-end tests for CiteWise search and answer CLIs.

These tests execute the real CLI pipeline (file loading, BM25, semantic cosine
scoring, Reciprocal Rank Fusion, prompt generation, citation resolution, and CLI formatting)
against the local sample reports and semantic index, mocking only the external
Gemini API boundary. Zero network or external API calls are made.
"""

import json
from pathlib import Path
import pytest

from citewise import answer_cli, hybrid_search_cli, search_cli, semantic_search_cli
from citewise.embedding_index import load_embedding_index


class FakeGeminiResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)


class FakeGeminiModels:
    def __init__(self, payload: dict):
        self.payload = payload
        self.contents_received = []

    def generate_content(self, model: str, contents: str, config: dict):
        self.contents_received.append(contents)
        return FakeGeminiResponse(self.payload)


class FakeGeminiClient:
    def __init__(self, payload: dict):
        self.models = FakeGeminiModels(payload)


@pytest.fixture(autouse=True)
def bypass_rate_limiter_for_e2e(monkeypatch):
    """Bypass the 21-second rate pacing sleep for instant offline testing."""
    import citewise.api_limits
    import citewise.answer_generator
    monkeypatch.setenv("CITEWISE_MIN_API_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(citewise.api_limits, "wait_for_api_slot", lambda: None)
    monkeypatch.setattr(citewise.answer_generator, "wait_for_api_slot", lambda: None)


@pytest.fixture
def sample_embedding_vector():
    """Load the real 768-dim embedding vector of the chunking passage from data/semantic_index.json."""
    index_path = Path("data/semantic_index.json")
    assert index_path.is_file(), "data/semantic_index.json must exist"
    data = load_embedding_index(index_path)
    # Find chunk b637c456e6e5-p2-c1 (Why Chunking Matters)
    for item in data["items"]:
        if "Chunking" in item["text"]:
            return list(item["embedding"])
    return list(data["items"][0]["embedding"])


def test_bm25_search_cli_e2e(capsys):
    """BM25 search CLI runs completely offline on output/ reports."""
    ret = search_cli.main() if hasattr(search_cli, "_dummy") else None
    # Invoke search_cli with real argv
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["search_cli", "chunking", "--reports", "output", "--top-k", "2"]
        ret = search_cli.main()
        assert ret == 0
        out = capsys.readouterr().out
        assert "Query: chunking" in out
        assert "Matching passages:" in out
        assert "text_sample.pdf — Page 2" in out
    finally:
        sys.argv = orig_argv


def test_hybrid_search_cli_e2e(sample_embedding_vector, monkeypatch, capsys):
    """Hybrid search CLI runs real BM25 + real cosine similarity + real RRF."""
    import citewise.embeddings
    import sys

    # Mock only the external query embedding call, returning the sample vector
    monkeypatch.setattr(
        citewise.embeddings,
        "embed_query",
        lambda query, client=None: sample_embedding_vector,
    )

    orig_argv = sys.argv
    try:
        sys.argv = [
            "hybrid_search_cli",
            "Why does chunking matter?",
            "--reports", "output",
            "--index", "data/semantic_index.json",
            "--top-k", "3",
        ]
        ret = hybrid_search_cli.main()
        assert ret == 0
        out = capsys.readouterr().out
        assert "Hybrid search: Why does chunking matter?" in out
        assert "text_sample.pdf — page 2" in out
        assert "Hybrid score:" in out
    finally:
        sys.argv = orig_argv


def test_answer_cli_e2e_grounded_answer(sample_embedding_vector, monkeypatch, capsys):
    """Answer CLI executes the entire pipeline end-to-end, mocking only Gemini generate_content."""
    import citewise.embeddings
    import citewise.answer_generator
    import sys

    # Mock query embedding
    monkeypatch.setattr(
        citewise.embeddings,
        "embed_query",
        lambda query, client=None: sample_embedding_vector,
    )

    fake_client = FakeGeminiClient({
        "answerable": True,
        "answer": "Chunking splits long documents into smaller passages while preserving context.",
        "source_ids": ["S1"],
    })
    monkeypatch.setattr(
        citewise.answer_generator,
        "_get_client",
        lambda: fake_client,
    )

    orig_argv = sys.argv
    try:
        sys.argv = [
            "answer_cli",
            "Why is chunking useful for RAG?",
            "--reports", "output",
            "--index", "data/semantic_index.json",
            "--top-k", "3",
        ]
        ret = answer_cli.main()
        assert ret == 0
        out = capsys.readouterr().out

        # Verify CLI output formatting
        assert "Question:" in out
        assert "Why is chunking useful for RAG?" in out
        assert "Answer:" in out
        assert "Chunking splits long documents into smaller passages" in out
        assert "Sources:" in out
        assert "text_sample.pdf, page 2" in out

        # Verify real prompt construction sent to Gemini
        assert len(fake_client.models.contents_received) == 1
        prompt = fake_client.models.contents_received[0]
        assert "[S1] File: text_sample.pdf, Page: 2" in prompt
        assert "STRICT EVIDENCE RULES:" in prompt
    finally:
        sys.argv = orig_argv


def test_answer_cli_e2e_abstention(sample_embedding_vector, monkeypatch, capsys):
    """Answer CLI handles unanswerable queries by displaying canonical abstention without citations."""
    import citewise.embeddings
    import citewise.answer_generator
    import sys

    monkeypatch.setattr(
        citewise.embeddings,
        "embed_query",
        lambda query, client=None: sample_embedding_vector,
    )

    fake_client = FakeGeminiClient({
        "answerable": False,
        "answer": "I couldn't find enough information in the provided PDF passages to answer this question reliably.",
        "source_ids": [],
    })
    monkeypatch.setattr(
        citewise.answer_generator,
        "_get_client",
        lambda: fake_client,
    )

    orig_argv = sys.argv
    try:
        sys.argv = [
            "answer_cli",
            "What is the average rainfall in the Amazon rainforest?",
            "--reports", "output",
            "--index", "data/semantic_index.json",
        ]
        ret = answer_cli.main()
        assert ret == 0
        out = capsys.readouterr().out
        assert "couldn't find enough information in the provided PDF passages" in out
        assert "No supporting PDF citations available." in out
    finally:
        sys.argv = orig_argv


def test_answer_cli_e2e_error_handling(monkeypatch, capsys):
    """Answer CLI handles invalid inputs and index errors gracefully."""
    import sys

    orig_argv = sys.argv
    try:
        # Missing index
        sys.argv = [
            "answer_cli",
            "Valid question?",
            "--reports", "output",
            "--index", "data/nonexistent_index.json",
        ]
        ret = answer_cli.main()
        assert ret == 1
        err = capsys.readouterr().err
        assert "semantic index not found" in err

        # Missing reports directory
        sys.argv = [
            "answer_cli",
            "Valid question?",
            "--reports", "nonexistent_dir",
            "--index", "data/semantic_index.json",
        ]
        ret = answer_cli.main()
        assert ret == 1
        err = capsys.readouterr().err
        assert "reports directory not found" in err
    finally:
        sys.argv = orig_argv


def test_semantic_search_cli_e2e_success(sample_embedding_vector, monkeypatch, capsys):
    """Semantic search CLI runs real index loading and cosine similarity, mocking only embed_query."""
    import citewise.embeddings
    import sys

    monkeypatch.setattr(
        citewise.embeddings,
        "embed_query",
        lambda query, client=None: sample_embedding_vector,
    )

    orig_argv = sys.argv
    try:
        sys.argv = [
            "semantic_search_cli",
            "Why does chunking matter?",
            "--index", "data/semantic_index.json",
            "--top-k", "2",
        ]
        ret = semantic_search_cli.main()
        assert ret == 0
        out = capsys.readouterr().out
        assert "Semantic search: Why does chunking matter?" in out
        assert "Results: 2" in out
        assert "text_sample.pdf — page 2" in out
        assert "Similarity:" in out
        assert "b637c456e6e5-p2-c1" in out
    finally:
        sys.argv = orig_argv


def test_semantic_search_cli_e2e_missing_index(capsys):
    """Semantic search CLI returns 1 and error message on stderr if index is missing."""
    import sys

    orig_argv = sys.argv
    try:
        sys.argv = [
            "semantic_search_cli",
            "Any query",
            "--index", "data/nonexistent_index.json",
        ]
        ret = semantic_search_cli.main()
        assert ret == 1
        err = capsys.readouterr().err
        assert "Error: semantic index not found" in err
        assert "nonexistent_index.json" in err
    finally:
        sys.argv = orig_argv


def test_semantic_search_cli_e2e_invalid_top_k(capsys):
    """Semantic search CLI raises SystemExit(2) on invalid top_k (< 1)."""
    import sys

    orig_argv = sys.argv
    try:
        sys.argv = [
            "semantic_search_cli",
            "Any query",
            "--top-k", "0",
        ]
        with pytest.raises(SystemExit) as exc_info:
            semantic_search_cli.main()
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--top-k must be at least 1" in err
    finally:
        sys.argv = orig_argv

