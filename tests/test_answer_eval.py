"""Unit tests for deterministic answer and citation evaluation."""

import pytest

from citewise.answer_eval import (
    compute_citation_metrics,
    compute_concept_coverage,
    compute_document_coverage,
    evaluate_answers,
    evaluate_single_answer,
    is_abstention,
)
from citewise.answer_generator import INSUFFICIENT_EVIDENCE
from citewise.eval_dataset import EvalCase, ExpectedCitation


def test_is_abstention():
    assert is_abstention(INSUFFICIENT_EVIDENCE, []) is True
    assert (
        is_abstention(
            "I couldn't find enough information in the provided passages.", []
        )
        is True
    )
    # Empty citations is treated as abstention even if answer text exists
    assert is_abstention("Some answer", []) is True
    # Non-empty answer with citations is not abstention
    assert (
        is_abstention(
            "CiteWise is a PDF RAG system.",
            [{"filename": "a.pdf", "page_number": 1, "chunk_id": "c1"}],
        )
        is False
    )


def test_compute_concept_coverage():
    req = ["PyMuPDF", "Tesseract", "1000 characters"]
    ans = "CiteWise uses PyMuPDF for extraction and Tesseract for OCR with 1000 characters chunk size."
    cov, missing = compute_concept_coverage(ans, req)
    assert cov == 1.0
    assert missing == []

    partial_ans = "CiteWise uses PyMuPDF for text extraction."
    cov, missing = compute_concept_coverage(partial_ans, req)
    assert cov == pytest.approx(1 / 3)
    assert missing == ["Tesseract", "1000 characters"]

    # Empty required concepts returns 1.0
    assert compute_concept_coverage("anything", []) == (1.0, [])


def test_compute_citation_metrics_answerable():
    gold = [
        ExpectedCitation(filename="a.pdf", page_number=1, chunk_id="c1"),
        ExpectedCitation(filename="a.pdf", page_number=2, chunk_id="c2"),
    ]

    # Perfect citations
    cited = [
        {"filename": "a.pdf", "page_number": 1, "chunk_id": "c1"},
        {"filename": "a.pdf", "page_number": 2, "chunk_id": "c2"},
    ]
    m = compute_citation_metrics(cited, gold, expected_answerable=True)
    assert m["chunk_precision"] == 1.0
    assert m["chunk_recall"] == 1.0
    assert m["chunk_f1"] == 1.0

    # Partial recall (c1 cited, c2 missing, extra c3 cited)
    cited_partial = [
        {"filename": "a.pdf", "page_number": 1, "chunk_id": "c1"},
        {"filename": "a.pdf", "page_number": 3, "chunk_id": "c3"},
    ]
    m2 = compute_citation_metrics(cited_partial, gold, expected_answerable=True)
    assert m2["chunk_precision"] == 0.5
    assert m2["chunk_recall"] == 0.5
    assert m2["chunk_f1"] == 0.5


def test_compute_citation_metrics_unanswerable():
    gold = []

    # Correct abstention: 0 citations produced
    m = compute_citation_metrics([], gold, expected_answerable=False)
    assert m["chunk_precision"] == 1.0
    assert m["chunk_recall"] == 1.0

    # Erroneous citations on unanswerable question
    bad_cited = [{"filename": "a.pdf", "page_number": 1, "chunk_id": "c1"}]
    m_bad = compute_citation_metrics(bad_cited, gold, expected_answerable=False)
    assert m_bad["chunk_precision"] == 0.0
    assert m_bad["chunk_recall"] == 0.0


def test_evaluate_single_answer_groundedness_check():
    case = EvalCase(
        id="Q1",
        question="What is A?",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="A is valid.",
        required_concepts=["valid"],
        expected_citations=[
            ExpectedCitation(filename="a.pdf", page_number=1, chunk_id="cA")
        ],
    )

    supplied_passages = [
        {"chunk_id": "cA", "filename": "a.pdf", "page_number": 1, "text": "A is valid"}
    ]

    # Correct citation that WAS supplied
    good_result = {
        "answer": "A is valid.",
        "citations": [{"filename": "a.pdf", "page_number": 1, "chunk_id": "cA"}],
    }
    eval_good = evaluate_single_answer(case, good_result, supplied_passages)
    assert eval_good.is_grounded_in_supplied is True
    assert eval_good.correct_decision is True
    assert eval_good.concept_coverage == 1.0

    # Hallucinated citation that was NOT in supplied passages
    hallucinated_result = {
        "answer": "A is valid.",
        "citations": [{"filename": "a.pdf", "page_number": 1, "chunk_id": "cUNSUPPLIED"}],
    }
    eval_bad = evaluate_single_answer(case, hallucinated_result, supplied_passages)
    assert eval_bad.is_grounded_in_supplied is False
    assert len(eval_bad.unsupplied_citations) == 1


def test_evaluate_answers_batch_workflow():
    cases = [
        EvalCase(
            id="Q1",
            question="What is X?",
            category="direct_factual",
            expected_answerable=True,
            reference_answer="X is cool",
            required_concepts=["cool"],
            expected_citations=[
                ExpectedCitation(filename="x.pdf", page_number=1, chunk_id="cX")
            ],
        ),
        EvalCase(
            id="Q2",
            question="What is unknown?",
            category="unanswerable",
            expected_answerable=False,
            reference_answer=INSUFFICIENT_EVIDENCE,
            required_concepts=[],
            expected_citations=[],
        ),
    ]

    results = {
        "Q1": {
            "answer": "X is very cool.",
            "citations": [{"filename": "x.pdf", "page_number": 1, "chunk_id": "cX"}],
        },
        "Q2": {
            "answer": INSUFFICIENT_EVIDENCE,
            "citations": [],
        },
    }

    passages = {
        "Q1": [{"chunk_id": "cX", "filename": "x.pdf", "page_number": 1, "text": "X is cool"}],
        "Q2": [],
    }

    report = evaluate_answers(cases, results, passages)
    assert report.total_cases == 2
    assert report.answerable_cases == 1
    assert report.unanswerable_cases == 1
    assert report.answerability_accuracy == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.mean_concept_coverage == 1.0
    assert report.mean_citation_precision == 1.0
    assert report.mean_citation_recall == 1.0
    assert report.grounded_citation_rate == 1.0
    assert report.error_count == 0


def test_answerability_classification_scenarios():
    ans_case = EvalCase(
        id="C1",
        question="What is Q?",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="Answer",
        required_concepts=["Answer"],
        expected_citations=[ExpectedCitation("a.pdf", 1, "c1")],
    )
    unans_case = EvalCase(
        id="C2",
        question="Unanswerable?",
        category="unanswerable",
        expected_answerable=False,
        reference_answer=INSUFFICIENT_EVIDENCE,
        required_concepts=[],
        expected_citations=[],
    )

    passages = [{"chunk_id": "c1", "filename": "a.pdf", "page_number": 1, "text": "Answer"}]

    # Scenario A: False negative (model abstains on answerable question)
    fn_res = evaluate_single_answer(
        ans_case,
        {"answer": INSUFFICIENT_EVIDENCE, "citations": []},
        passages,
    )
    assert fn_res.model_answerable is False
    assert fn_res.correct_decision is False
    assert fn_res.concept_coverage == 0.0

    # Scenario B: False positive (model attempts answer on unanswerable question)
    fp_res = evaluate_single_answer(
        unans_case,
        {"answer": "Hallucinated answer", "citations": [{"chunk_id": "c1", "filename": "a.pdf", "page_number": 1}]},
        passages,
    )
    assert fp_res.model_answerable is True
    assert fp_res.correct_decision is False
    assert fp_res.correct_abstention is False
    assert fp_res.concept_coverage == 0.0
    assert fp_res.citation_precision == 0.0

    # Scenario C: True negative (model correctly abstains on unanswerable question)
    tn_res = evaluate_single_answer(
        unans_case,
        {"answer": INSUFFICIENT_EVIDENCE, "citations": []},
        [],
    )
    assert tn_res.model_answerable is False
    assert tn_res.correct_decision is True
    assert tn_res.correct_abstention is True
    assert tn_res.concept_coverage == 1.0
    assert tn_res.citation_precision == 1.0


def test_page_vs_chunk_citation_matching():
    case = EvalCase(
        id="C1",
        question="Q?",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="A",
        required_concepts=[],
        expected_citations=[
            ExpectedCitation(filename="report.pdf", page_number=5, chunk_id="chunk-correct")
        ],
    )
    passages = [
        {"chunk_id": "chunk-neighbor", "filename": "report.pdf", "page_number": 5, "text": "A"}
    ]

    # Cited the correct page (5) of report.pdf, but a different chunk ID (chunk-neighbor)
    gen = {
        "answer": "A",
        "citations": [
            {"filename": "report.pdf", "page_number": 5, "chunk_id": "chunk-neighbor"}
        ],
    }
    res = evaluate_single_answer(case, gen, passages)
    assert res.citation_precision == 0.0
    assert res.citation_recall == 0.0
    assert res.page_precision == 1.0
    assert res.page_recall == 1.0
    assert res.is_grounded_in_supplied is True


def test_malformed_responses_handled_cleanly():
    case = EvalCase(
        id="C1",
        question="Q?",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="A",
        required_concepts=["term"],
        expected_citations=[ExpectedCitation("a.pdf", 1, "c1")],
    )

    # 1. Non-dict generated_result
    res1 = evaluate_single_answer(case, "just a string", [])
    assert res1.is_error is True
    assert "Malformed result" in res1.error_message

    # 2. Citations is None or not a list
    res2 = evaluate_single_answer(case, {"answer": "term exists", "citations": None}, [])
    assert res2.citation_precision == 0.0
    assert res2.citation_recall == 0.0

    # 3. Answer is non-string or None
    res3 = evaluate_single_answer(case, {"answer": None, "citations": []}, [])
    assert res3.concept_coverage == 0.0


def test_generation_error_tracking():
    cases = [
        EvalCase(
            id="C1",
            question="Q?",
            category="direct_factual",
            expected_answerable=True,
            reference_answer="A",
            required_concepts=[],
            expected_citations=[],
        )
    ]
    results = {
        "C1": {
            "answer": "",
            "citations": [],
            "is_error": True,
            "error_message": "Rate limit 429",
        }
    }
    report = evaluate_answers(cases, results, {"C1": []})
    assert report.error_count == 1
    assert report.answerability_accuracy == 0.0
    assert report.per_case_results[0].is_error is True
    assert report.per_case_results[0].error_message == "Rate limit 429"


def test_sample_answer_fixtures_json():
    import json
    from pathlib import Path

    fixture_path = Path("tests/fixtures/sample_answer_fixtures.json")
    assert fixture_path.is_file()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    responses = payload["responses"]

    assert "EVAL-001" in responses
    assert "EVAL-017" in responses
    assert is_abstention(responses["EVAL-017"]["answer"], responses["EVAL-017"]["citations"]) is True


def test_compute_document_coverage_scenarios():
    gold = [
        ExpectedCitation(filename="nist_ai_rmf.pdf", page_number=12, chunk_id="c1"),
        ExpectedCitation(filename="nist_csf_2.pdf", page_number=30, chunk_id="c2"),
    ]

    # 1. Full coverage: cites both required documents
    cited_full = [
        {"filename": "nist_ai_rmf.pdf", "page_number": 12, "chunk_id": "c1"},
        {"filename": "nist_csf_2.pdf", "page_number": 30, "chunk_id": "c2"},
    ]
    cov_full = compute_document_coverage(cited_full, gold, expected_answerable=True)
    assert cov_full == 1.0

    # 2. Partial coverage: cites only one of two required documents
    cited_partial = [
        {"filename": "nist_ai_rmf.pdf", "page_number": 12, "chunk_id": "c1"},
    ]
    cov_part = compute_document_coverage(cited_partial, gold, expected_answerable=True)
    assert cov_part == 0.5

    # 3. No coverage: cites a completely different document
    cited_none = [
        {"filename": "other.pdf", "page_number": 1, "chunk_id": "c9"},
    ]
    cov_none = compute_document_coverage(cited_none, gold, expected_answerable=True)
    assert cov_none == 0.0

    # 4. Irrelevant extra documents: cites both required plus an extra document
    cited_extra = [
        {"filename": "nist_ai_rmf.pdf", "page_number": 12, "chunk_id": "c1"},
        {"filename": "nist_csf_2.pdf", "page_number": 30, "chunk_id": "c2"},
        {"filename": "extra.pdf", "page_number": 5, "chunk_id": "c8"},
    ]
    cov_extra = compute_document_coverage(cited_extra, gold, expected_answerable=True)
    assert cov_extra == 1.0  # (2 covered / 2 required)

    # 5. Single-document case
    gold_single = [ExpectedCitation(filename="report.pdf", page_number=1, chunk_id="c1")]
    assert compute_document_coverage(
        [{"filename": "report.pdf", "page_number": 1, "chunk_id": "c1"}],
        gold_single,
        expected_answerable=True,
    ) == 1.0
    assert compute_document_coverage(
        [{"filename": "wrong.pdf", "page_number": 1, "chunk_id": "c2"}],
        gold_single,
        expected_answerable=True,
    ) == 0.0

    # 6. Unanswerable case returns None (not applicable)
    assert compute_document_coverage([], [], expected_answerable=False) is None
    assert compute_document_coverage(cited_full, gold, expected_answerable=False) is None

    # 7. Model abstention on answerable case returns 0.0
    assert compute_document_coverage(
        [], gold, expected_answerable=True, model_answerable=False
    ) == 0.0


def test_document_coverage_in_evaluate_answers_aggregation():
    case_cross = EvalCase(
        id="ADV-001",
        question="Compare docA and docB",
        category="cross_document",
        expected_answerable=True,
        reference_answer="Synthesis",
        required_concepts=["synthesis"],
        expected_citations=[
            ExpectedCitation("docA.pdf", 1, "cA"),
            ExpectedCitation("docB.pdf", 2, "cB"),
        ],
    )
    case_single = EvalCase(
        id="ADV-002",
        question="Fact in docA",
        category="direct_factual",
        expected_answerable=True,
        reference_answer="Fact",
        required_concepts=["fact"],
        expected_citations=[
            ExpectedCitation("docA.pdf", 1, "cA"),
        ],
    )
    case_unans = EvalCase(
        id="ADV-003",
        question="Unanswerable",
        category="unanswerable",
        expected_answerable=False,
        reference_answer="None",
        required_concepts=[],
        expected_citations=[],
    )

    results = {
        # ADV-001: cites only docA (0.5 doc coverage)
        "ADV-001": {
            "answer": "synthesis",
            "citations": [{"filename": "docA.pdf", "page_number": 1, "chunk_id": "cA"}],
        },
        # ADV-002: cites docA (1.0 doc coverage)
        "ADV-002": {
            "answer": "fact",
            "citations": [{"filename": "docA.pdf", "page_number": 1, "chunk_id": "cA"}],
        },
        # ADV-003: abstains (None doc coverage)
        "ADV-003": {
            "answer": "I couldn't find enough information in the provided PDF passages to answer this question reliably.",
            "citations": [],
        },
    }

    passages = {
        "ADV-001": [{"chunk_id": "cA", "filename": "docA.pdf", "page_number": 1}],
        "ADV-002": [{"chunk_id": "cA", "filename": "docA.pdf", "page_number": 1}],
        "ADV-003": [],
    }

    report = evaluate_answers([case_cross, case_single, case_unans], results, passages)
    assert report.per_case_results[0].document_coverage == 0.5
    assert report.per_case_results[1].document_coverage == 1.0
    assert report.per_case_results[2].document_coverage is None

    # Mean document coverage should average only applicable cases: (0.5 + 1.0) / 2 = 0.75
    assert report.mean_document_coverage == 0.75


