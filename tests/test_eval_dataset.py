"""Unit tests for evaluation dataset schema and validation."""

import json
from pathlib import Path
import pytest

from citewise.eval_dataset import (
    EvalCase,
    ExpectedCitation,
    validate_eval_dataset,
)


def test_eval_case_to_dict():
    case = EvalCase(
        id="TEST-001",
        question="What is test?",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="This is a test.",
        required_concepts=["test"],
        expected_citations=[
            ExpectedCitation(filename="test.pdf", page_number=1, chunk_id="c1")
        ],
        difficulty="easy",
        rationale="For testing",
    )
    d = case.to_dict()
    assert d["id"] == "TEST-001"
    assert d["category"] == "direct_factual"
    assert len(d["expected_citations"]) == 1
    assert d["expected_citations"][0]["chunk_id"] == "c1"


def test_validate_eval_dataset_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        validate_eval_dataset(tmp_path / "nonexistent.json")


def test_validate_eval_dataset_invalid_json(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not-valid-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        validate_eval_dataset(bad_file)


def test_validate_eval_dataset_empty_list(tmp_path: Path):
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_eval_dataset(empty_file)


def test_validate_eval_dataset_duplicate_id(tmp_path: Path):
    data = [
        {
            "id": "DUP-1",
            "question": "Q1",
            "category": "direct_factual",
            "expected_answerable": False,
            "reference_answer": "None",
        },
        {
            "id": "DUP-1",
            "question": "Q2",
            "category": "direct_factual",
            "expected_answerable": False,
            "reference_answer": "None",
        },
    ]
    f = tmp_path / "dup.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate case id"):
        validate_eval_dataset(f)


def test_validate_eval_dataset_invalid_category(tmp_path: Path):
    data = [
        {
            "id": "CAT-1",
            "question": "Q1",
            "category": "invalid_category",
            "expected_answerable": False,
            "reference_answer": "None",
        }
    ]
    f = tmp_path / "cat.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid category"):
        validate_eval_dataset(f)


def test_validate_eval_dataset_unanswerable_with_citations(tmp_path: Path):
    data = [
        {
            "id": "UNANS-1",
            "question": "Q1",
            "category": "unanswerable",
            "expected_answerable": False,
            "reference_answer": "None",
            "expected_citations": [
                {"filename": "test.pdf", "page_number": 1, "chunk_id": "c1"}
            ],
        }
    ]
    f = tmp_path / "unans.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="is unanswerable but provides expected citations"):
        validate_eval_dataset(f)


def test_validate_eval_dataset_answerable_without_citations(tmp_path: Path):
    data = [
        {
            "id": "ANS-1",
            "question": "Q1",
            "category": "direct_factual",
            "expected_answerable": True,
            "reference_answer": "Answer",
            "expected_citations": [],
        }
    ]
    f = tmp_path / "ans.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="provides no expected citations"):
        validate_eval_dataset(f)


def test_validate_eval_dataset_answerable_without_chunk_id(tmp_path: Path):
    data = [
        {
            "id": "ANS-2",
            "question": "Q2",
            "category": "direct_factual",
            "expected_answerable": True,
            "reference_answer": "Answer",
            "expected_citations": [{"filename": "test.pdf", "page_number": 1}],
        }
    ]
    f = tmp_path / "ans2.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="provides no citation with a valid chunk_id"):
        validate_eval_dataset(f)


def test_validate_real_dataset_against_reports():
    cases = validate_eval_dataset(
        "evaluation/dataset.json",
        reports_dir="evaluation/reports",
    )
    assert len(cases) == 22
    categories = {c.category for c in cases}
    assert "direct_factual" in categories
    assert "unanswerable" in categories
    assert "paraphrased_semantic" in categories
    assert "cross_document" in categories

