"""Unit and regression tests for the local pre-generation evidence gate."""

import json
from pathlib import Path
import pytest

from citewise.evidence_gate import (
    EvidenceGateResult,
    evaluate_evidence_gate,
    extract_content_tokens,
)
from citewise.retriever import BM25Retriever, load_chunks


def test_extract_content_tokens_filters_stopwords():
    text = "What is the primary function of risk management in 2024?"
    tokens = extract_content_tokens(text)
    assert "what" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "in" not in tokens
    assert "primary" in tokens
    assert "function" in tokens
    assert "risk" in tokens
    assert "management" in tokens
    assert "2024" in tokens


def test_evidence_gate_empty_query():
    result = evaluate_evidence_gate("", [{"text": "Sample text", "score": 10.0}])
    assert result.decision == "reject"
    assert result.reason == "Empty query"
    assert not result.should_generate


def test_evidence_gate_empty_passages():
    result = evaluate_evidence_gate("What is NIST?", [])
    assert result.decision == "reject"
    assert result.reason == "Empty retrieval"
    assert not result.should_generate


def test_evidence_gate_stopwords_only_query():
    result = evaluate_evidence_gate(
        "What is it?",
        [{"text": "It is a framework.", "score": 2.0}],
        retriever_type="bm25",
    )
    assert result.decision == "borderline"
    assert result.should_generate


def test_evidence_gate_bm25_zero_lexical_overlap_reject():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "notes.pdf",
            "page_number": 1,
            "score": 4.5,
            "text": "The internet protocol suite is the conceptual model for computer network communications.",
        }
    ]
    # Amazon rainforest has zero content tokens in the passage
    result = evaluate_evidence_gate(
        "What is the average rainfall in the Amazon rainforest?",
        passages,
        retriever_type="bm25",
    )
    assert result.decision == "reject"
    assert result.reason == "Zero lexical overlap"
    assert not result.should_generate
    assert result.metrics["matched_terms_count"] == 0
    assert result.metrics["lexical_overlap_ratio"] == 0.0


def test_evidence_gate_bm25_strong_pass():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "nist_csf.pdf",
            "page_number": 8,
            "score": 24.5,
            "text": "The six Functions in the NIST CSF 2.0 Core are GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, and RECOVER.",
        }
    ]
    result = evaluate_evidence_gate(
        "What are the six Functions that organize outcomes in NIST CSF 2.0 Core?",
        passages,
        retriever_type="bm25",
    )
    assert result.decision == "pass"
    assert result.reason in ("Strong lexical support", "Substantial query term overlap")
    assert result.should_generate
    assert result.metrics["lexical_overlap_ratio"] >= 0.5


def test_evidence_gate_bm25_low_score_low_overlap_reject():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "page_number": 1,
            "score": 2.1,
            "text": "This document contains general administrative information for all employees.",
        }
    ]
    # Only "administrative" matches, score is 2.1 < 5.0, overlap is 1/6 < 0.20
    query = "What are the maximum administrative fines in euros established by the European Union AI Act?"
    result = evaluate_evidence_gate(query, passages, retriever_type="bm25")
    assert result.decision == "reject"
    assert result.reason == "Insufficient lexical relevance"
    assert not result.should_generate


def test_evidence_gate_bm25_borderline():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "page_number": 2,
            "score": 7.0,
            "text": "Organizations establish governance tiers to oversee policy.",
        }
    ]
    query = "Explain cybersecurity governance tiers and management rigor."
    result = evaluate_evidence_gate(query, passages, retriever_type="bm25")
    assert result.decision in ("pass", "borderline")
    assert result.should_generate


def test_evidence_gate_semantic_high_similarity_pass():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "nist_ai.pdf",
            "page_number": 25,
            "score": 0.78,
            "text": "The AI RMF Core comprises four functions: GOVERN, MAP, MEASURE, and MANAGE.",
        }
    ]
    result = evaluate_evidence_gate(
        "What four functions make up the AI RMF Core?",
        passages,
        retriever_type="semantic",
    )
    assert result.decision == "pass"
    assert result.reason == "High semantic confidence"
    assert result.should_generate
    assert result.metrics["semantic_similarity"] == 0.78


def test_evidence_gate_semantic_zero_overlap_low_similarity_reject():
    passages = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "page_number": 1,
            "score": 0.21,
            "text": "The quick brown fox jumps over the lazy dog.",
        }
    ]
    result = evaluate_evidence_gate(
        "In what year did Guido van Rossum release the Python programming language?",
        passages,
        retriever_type="semantic",
    )
    assert result.decision == "reject"
    assert result.reason == "Uncorrelated retrieval"
    assert not result.should_generate


def test_evidence_gate_hybrid_paraphrase_preserves_generation():
    # Paraphrased question where exact vocabulary differs, but semantic similarity is high
    passages = [
        {
            "chunk_id": "c1",
            "filename": "nist_ai.pdf",
            "page_number": 22,
            "score": 0.72,
            "semantic_score": 0.72,
            "bm25_score": 1.2,
            "text": "Fairness in AI includes managing harmful bias and addressing demographic disparities through multidisciplinary teams.",
        }
    ]
    result = evaluate_evidence_gate(
        "How can algorithms prevent disparate adverse impacts across demographics?",
        passages,
        retriever_type="hybrid",
    )
    # The gate must NOT reject valid paraphrases
    assert result.decision in ("pass", "borderline")
    assert result.should_generate


def test_evidence_gate_dataset_zero_false_rejections():
    """Verify that no answerable benchmark query is falsely rejected under BM25."""
    dataset_path = Path("evaluation/dataset.json")
    if not dataset_path.is_file():
        pytest.skip("Dataset file not found.")

    report_files = sorted(Path("evaluation/reports").glob("*.json"))
    if not report_files:
        pytest.skip("No reports available.")

    chunks = load_chunks(report_files)
    retriever = BM25Retriever(chunks=chunks)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    answerable_cases = [case for case in dataset if case.get("expected_answerable") is True]
    false_rejections = []

    for case in answerable_cases:
        passages = retriever.search(case["question"], top_k=5)
        result = evaluate_evidence_gate(case["question"], passages, retriever_type="bm25")
        if result.decision == "reject":
            false_rejections.append((case["id"], result.reason, result.explanation))

    assert false_rejections == [], f"Found false rejections in benchmark: {false_rejections}"


def test_evidence_gate_combined_multi_passage_evidence():
    """Verify that multiple passages jointly covering query terms result in a pass with combined evidence metric."""
    passages = [
        {
            "chunk_id": "c1",
            "filename": "web_tech.pdf",
            "page_number": 3,
            "score": 6.0,
            "text": "There are three levels of style sheets: inline, document level, and external style sheets.",
        },
        {
            "chunk_id": "c2",
            "filename": "web_tech.pdf",
            "page_number": 24,
            "score": 5.5,
            "text": "JavaScript was originated with Netscape and it is invented by Brendan Eich.",
        },
    ]
    query = "What are the style sheet levels and who invented JavaScript?"
    result = evaluate_evidence_gate(query, passages, retriever_type="bm25")

    assert result.decision == "pass"
    assert result.should_generate is True
    assert result.metrics["combined_evidence"] is True
    assert result.reason == "Combined multi-passage evidence"
    assert "jointly supported" in result.explanation


def test_evidence_gate_semantic_similarity_unavailable_represented_as_none():
    """Verify that missing semantic similarity is represented as None/NA and never as 0.0."""
    passages = [
        {
            "chunk_id": "c1",
            "filename": "doc.pdf",
            "page_number": 1,
            "score": 0.0,
            "bm25_score": 0.0,
            "semantic_score": None,
            "text": "Completely unrelated content about planetary orbits.",
        }
    ]
    query = "What is the capital of France?"
    result = evaluate_evidence_gate(query, passages, retriever_type="hybrid")

    assert result.decision == "reject"
    assert result.metrics["semantic_similarity"] is None
    assert "N/A" in result.explanation
    assert not result.should_generate


def test_evidence_gate_vocabulary_overlap_unsupported_fact_gate_policy():
    """Verify that when query words overlap with PDF text but ask an unsupported fact,
    the gate allows generation so Gemini's grounded abstention can evaluate factuality."""
    passages = [
        {
            "chunk_id": "c24",
            "filename": "web_tech.pdf",
            "page_number": 24,
            "score": 6.0,
            "text": "JavaScript was originated with Netscape and it is invented by Brendan Eich.",
        }
    ]
    # "JavaScript", "invented", "Brendan", "Eich" overlap, but "year" or "1995" is absent
    query = "In what exact calendar year was JavaScript invented by Brendan Eich?"
    result = evaluate_evidence_gate(query, passages, retriever_type="bm25")

    # Documented conservative policy: lexical overlap >= 50% forwards to generator
    assert result.decision in ("pass", "borderline")
    assert result.should_generate is True


def test_evidence_gate_paraphrase_preserves_generation():
    """Verify that a paraphrased query with modest word overlap forwards to generator."""
    passages = [
        {
            "chunk_id": "c10",
            "filename": "policy.pdf",
            "page_number": 5,
            "score": 7.5,
            "text": "Supervisory executives oversee organizational risk governance frameworks.",
        }
    ]
    query = "How do senior leaders monitor cybersecurity risk strategies?"
    result = evaluate_evidence_gate(query, passages, retriever_type="bm25")

    # Gate should not reject - conservative policy forwards to generator
    assert result.decision in ("pass", "borderline")
    assert result.should_generate is True

