"""Unit tests for the evaluate_cli interface."""

import json
from pathlib import Path
import pytest

from citewise.evaluate_cli import main


def test_evaluate_cli_offline_retrieval_success(tmp_path: Path):
    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "retrieval",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(results_dir),
        "--top-k", "3",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 1
    assert len(md_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["metadata"]["mode"] == "retrieval"
    assert payload["metadata"]["retriever"] == "bm25"
    assert payload["retrieval"]["total_cases"] == 22
    assert "mean_recall_at_3" in payload["retrieval"]


def test_evaluate_cli_offline_full_mode_success(tmp_path: Path):
    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "full",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    assert len(json_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["metadata"]["mode"] == "full"
    assert payload["retrieval"]["total_cases"] == 2
    assert payload["answer"]["total_cases"] == 2


def test_evaluate_cli_live_mode_cap_enforcement(tmp_path: Path, capsys):
    # Live mode without max-cases <= 3 must fail fast
    ret = main([
        "--mode", "retrieval",
        "--live",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(tmp_path),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "strictly capped at 3 cases" in captured.err


def test_evaluate_cli_live_mode_excess_max_cases(tmp_path: Path, capsys):
    ret = main([
        "--mode", "retrieval",
        "--live",
        "--max-cases", "5",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(tmp_path),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "strictly capped at 3 cases" in captured.err


def test_evaluate_cli_invalid_reports_dir(tmp_path: Path, capsys):
    ret = main([
        "--reports", str(tmp_path / "nonexistent"),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Reports directory not found" in captured.err


def test_evaluate_cli_semantic_offline_missing_cache_fails_fast(tmp_path: Path, monkeypatch, capsys):
    def unexpected_api_call(*args, **kwargs):
        raise AssertionError("Gemini API should NOT be called in offline mode!")

    import citewise.embeddings
    monkeypatch.setattr(citewise.embeddings, "embed_query", unexpected_api_call)

    cache_path = tmp_path / "nonexistent_cache.json"
    results_dir = tmp_path / "results"

    ret = main([
        "--mode", "retrieval",
        "--retriever", "semantic",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Query embedding cache miss" in captured.err
    assert "--allow-query-embedding-api" in captured.err


def test_evaluate_cli_hybrid_offline_missing_cache_fails_fast(tmp_path: Path, monkeypatch, capsys):
    def unexpected_api_call(*args, **kwargs):
        raise AssertionError("Gemini API should NOT be called in offline mode!")

    import citewise.embeddings
    monkeypatch.setattr(citewise.embeddings, "embed_query", unexpected_api_call)

    cache_path = tmp_path / "nonexistent_cache.json"
    results_dir = tmp_path / "results"

    ret = main([
        "--mode", "retrieval",
        "--retriever", "hybrid",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Query embedding cache miss" in captured.err
    assert "--allow-query-embedding-api" in captured.err


def test_evaluate_cli_offline_with_populated_cache_zero_api_calls(tmp_path: Path, monkeypatch):
    from citewise.embeddings import EMBEDDING_DIMENSIONS, MODEL_NAME

    def unexpected_api_call(*args, **kwargs):
        raise AssertionError("Zero API calls allowed when cache is populated!")

    import citewise.embeddings
    monkeypatch.setattr(citewise.embeddings, "embed_query", unexpected_api_call)

    # Read first two questions from dataset
    raw_dataset = json.loads(Path("evaluation/dataset.json").read_text(encoding="utf-8"))
    cases = raw_dataset if isinstance(raw_dataset, list) else raw_dataset["cases"]
    q1 = cases[0]["question"]
    q2 = cases[1]["question"]

    dummy_vector = [0.0] * EMBEDDING_DIMENSIONS
    dummy_vector[0] = 0.5

    cache_path = tmp_path / "populated_cache.json"
    cache_path.write_text(
        json.dumps({
            "version": 1,
            "model": MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "queries": {
                q1: dummy_vector,
                q2: dummy_vector,
            },
        }),
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"

    # Semantic evaluation succeeds completely offline
    ret_semantic = main([
        "--mode", "retrieval",
        "--retriever", "semantic",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret_semantic == 0

    # Hybrid evaluation succeeds completely offline sharing the exact same cache
    ret_hybrid = main([
        "--mode", "retrieval",
        "--retriever", "hybrid",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret_hybrid == 0


def test_evaluate_cli_allow_query_embedding_api_populates_and_shares(tmp_path: Path, monkeypatch):
    from citewise.embeddings import EMBEDDING_DIMENSIONS

    api_calls = []

    def mock_embed(query: str, client=None):
        api_calls.append(query)
        vec = [0.0] * EMBEDDING_DIMENSIONS
        vec[0] = 0.77
        return vec

    import citewise.embeddings
    monkeypatch.setattr(citewise.embeddings, "embed_query", mock_embed)

    cache_path = tmp_path / "new_query_cache.json"
    results_dir = tmp_path / "results"

    # Step 1: Run semantic with --allow-query-embedding-api
    ret1 = main([
        "--mode", "retrieval",
        "--retriever", "semantic",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--allow-query-embedding-api",
        "--max-cases", "2",
    ])
    assert ret1 == 0
    assert len(api_calls) == 2
    assert cache_path.exists()

    # Step 2: Run hybrid WITHOUT --allow-query-embedding-api (offline) using the same cache
    # It must make ZERO additional API calls because embeddings were cached in Step 1
    ret2 = main([
        "--mode", "retrieval",
        "--retriever", "hybrid",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(cache_path),
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret2 == 0
    # No new API calls
    assert len(api_calls) == 2


def test_evaluate_cli_corrupted_cache_handled_cleanly(tmp_path: Path, capsys):
    corrupted_cache = tmp_path / "corrupted.json"
    corrupted_cache.write_text("{ not json ", encoding="utf-8")

    ret = main([
        "--mode", "retrieval",
        "--retriever", "semantic",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--index", "evaluation/data/semantic_index.json",
        "--query-cache", str(corrupted_cache),
        "--results-dir", str(tmp_path / "results"),
        "--max-cases", "2",
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Initialization Error" in captured.err


def test_evaluate_cli_offline_answer_mode_labels_mock_simulation(tmp_path: Path, monkeypatch):
    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generate_answer should NOT be called in offline mode!")

    import citewise.evaluate_cli as cli_module
    monkeypatch.setattr(cli_module, "generate_answer", unexpected_generate)

    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "answer",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 1
    assert len(md_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["metadata"]["is_mock_generation"] is True
    assert payload["metadata"]["generator_type"] == "offline_simulation_mock"
    assert payload["answer"]["is_mock_generation"] is True
    assert "Offline simulation run" in payload["answer"]["simulation_notice"]

    md_content = md_reports[0].read_text(encoding="utf-8")
    assert "**Mock Generation**: `True`" in md_content


def test_evaluate_cli_mock_responses_fixture_offline(tmp_path: Path, monkeypatch):
    def unexpected_generate(*args, **kwargs):
        raise AssertionError("generate_answer should NOT be called when using mock fixture!")

    import citewise.evaluate_cli as cli_module
    monkeypatch.setattr(cli_module, "generate_answer", unexpected_generate)

    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "answer",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--mock-responses", "tests/fixtures/sample_answer_fixtures.json",
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    assert len(json_reports) == 1
    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["metadata"]["generator_type"] == "fixture_mock"
    assert payload["metadata"]["is_mock_generation"] is True
    assert payload["answer"]["is_mock_generation"] is True


def test_evaluate_cli_report_preserves_answer_and_citations(tmp_path: Path):
    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "answer",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--mock-responses", "tests/fixtures/sample_answer_fixtures.json",
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 1
    assert len(md_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    per_case = payload["answer"]["per_case"]
    assert len(per_case) == 2

    case_0 = per_case[0]
    assert case_0["case_id"] == "EVAL-001"
    assert "actual_answer" in case_0
    assert len(case_0["actual_answer"]) > 0
    assert "citations" in case_0
    assert isinstance(case_0["citations"], list)
    assert "source_ids" in case_0
    assert case_0["question"] == "What are the four primary functions that comprise the NIST AI Risk Management Framework Core?"

    md_content = md_reports[0].read_text(encoding="utf-8")
    assert "## Detailed Case Inspection" in md_content
    assert "### `EVAL-001`" in md_content
    assert "- **Generated Answer**:" in md_content


def test_evaluate_cli_report_serializes_document_coverage(tmp_path: Path):
    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "answer",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--mock-responses", "tests/fixtures/sample_answer_fixtures.json",
        "--results-dir", str(results_dir),
        "--max-cases", "2",
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert "mean_document_coverage" in payload["answer"]
    assert payload["answer"]["mean_document_coverage"] is not None

    per_case = payload["answer"]["per_case"]
    assert "document_coverage" in per_case[0]
    assert per_case[0]["document_coverage"] == 1.0

    md_content = md_reports[0].read_text(encoding="utf-8")
    assert "| **Mean Document Coverage** |" in md_content
    assert "DocCoverage=" in md_content


def test_evaluate_cli_inapplicable_document_coverage_renders_na(tmp_path: Path):
    # Create dataset with only unanswerable cases
    unans_dataset = tmp_path / "unans_dataset.json"
    unans_dataset.write_text(
        json.dumps([
            {
                "id": "UNANS-001",
                "question": "Unanswerable query?",
                "category": "unanswerable",
                "expected_answerable": False,
                "reference_answer": "None",
                "required_concepts": [],
                "expected_citations": [],
                "difficulty": "easy",
                "rationale": "testing",
            }
        ]),
        encoding="utf-8",
    )

    mock_fixtures = tmp_path / "mock_fixtures.json"
    mock_fixtures.write_text(
        json.dumps({
            "description": "mock unanswerable",
            "responses": {
                "UNANS-001": {
                    "answer": "I couldn't find enough information in the provided PDF passages to answer this question reliably.",
                    "citations": [],
                }
            },
        }),
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    ret = main([
        "--mode", "answer",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", str(unans_dataset),
        "--mock-responses", str(mock_fixtures),
        "--results-dir", str(results_dir),
    ])
    assert ret == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 1

    payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert payload["answer"]["mean_document_coverage"] is None
    assert payload["answer"]["per_case"][0]["document_coverage"] is None

    md_content = md_reports[0].read_text(encoding="utf-8")
    assert "| **Mean Document Coverage** | N/A |" in md_content
    assert "DocCoverage=N/A" in md_content


def test_evaluate_cli_unique_run_ids_no_overwriting(tmp_path: Path):
    results_dir = tmp_path / "results"
    import time

    ret1 = main([
        "--mode", "retrieval",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(results_dir),
        "--max-cases", "1",
    ])
    assert ret1 == 0

    time.sleep(1.1)

    ret2 = main([
        "--mode", "retrieval",
        "--retriever", "bm25",
        "--reports", "evaluation/reports",
        "--dataset", "evaluation/dataset.json",
        "--results-dir", str(results_dir),
        "--max-cases", "1",
    ])
    assert ret2 == 0

    json_reports = list(results_dir.glob("*.json"))
    md_reports = list(results_dir.glob("*.md"))
    assert len(json_reports) == 2
    assert len(md_reports) == 2
    assert json_reports[0].name != json_reports[1].name




