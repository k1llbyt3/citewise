"""Unit tests for retrieval evaluation metrics and evaluation engine."""

import math
import pytest

from citewise.eval_dataset import EvalCase, ExpectedCitation
from citewise.retrieval_eval import (
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_retrieval,
)


def test_compute_recall_at_k():
    gold = {"c1", "c2", "c3"}

    # Perfect recall in top 3
    assert compute_recall_at_k(["c1", "c2", "c3", "c4"], gold, k=3) == 1.0

    # Partial recall in top 1
    assert compute_recall_at_k(["c1", "c4", "c5"], gold, k=1) == pytest.approx(1 / 3)

    # Partial recall in top 2
    assert compute_recall_at_k(["c1", "c4", "c2"], gold, k=2) == pytest.approx(1 / 3)

    # Empty retrieved
    assert compute_recall_at_k([], gold, k=5) == 0.0

    # Empty gold
    assert compute_recall_at_k(["c1"], set(), k=5) == 0.0


def test_compute_reciprocal_rank():
    gold = {"target"}

    # Rank 1 hit
    rr, rank = compute_reciprocal_rank(["target", "other1", "other2"], gold)
    assert rr == 1.0
    assert rank == 1

    # Rank 3 hit
    rr, rank = compute_reciprocal_rank(["other1", "other2", "target"], gold)
    assert rr == pytest.approx(1 / 3)
    assert rank == 3

    # No hit
    rr, rank = compute_reciprocal_rank(["other1", "other2"], gold)
    assert rr == 0.0
    assert rank is None

    # Empty inputs
    assert compute_reciprocal_rank([], gold) == (0.0, None)
    assert compute_reciprocal_rank(["c1"], set()) == (0.0, None)


def test_compute_ndcg_at_k():
    gold = {"c1", "c2"}

    # Perfect ranking: both gold items at ranks 1 and 2
    ndcg = compute_ndcg_at_k(["c1", "c2", "c3"], gold, k=5)
    assert ndcg == pytest.approx(1.0)

    # Gold items at ranks 2 and 3
    # DCG = 1/log2(3) + 1/log2(4) = 1/1.58496 + 0.5 = 0.6309 + 0.5 = 1.1309
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # nDCG = 1.1309 / 1.6309 = 0.6934
    ndcg_lower = compute_ndcg_at_k(["c3", "c1", "c2"], gold, k=5)
    assert 0.0 < ndcg_lower < 1.0

    # No hits in top k
    assert compute_ndcg_at_k(["c3", "c4", "c5"], gold, k=5) == 0.0

    # Empty gold
    assert compute_ndcg_at_k(["c1"], set(), k=5) == 0.0


def test_compute_ndcg_with_graded_relevance():
    gold = {"c1", "c2"}
    relevance = {"c1": 2.0, "c2": 1.0}

    # Best possible order
    assert compute_ndcg_at_k(["c1", "c2"], gold, k=2, relevance_scores=relevance) == pytest.approx(1.0)

    # Inverted order
    ndcg_inverted = compute_ndcg_at_k(["c2", "c1"], gold, k=2, relevance_scores=relevance)
    assert ndcg_inverted < 1.0


class DummyRetriever:
    def __init__(self, responses: dict[str, list[str]]):
        self.responses = responses

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        ids = self.responses.get(query, [])[:top_k]
        return [{"chunk_id": cid, "score": 1.0} for cid in ids]


def test_evaluate_retrieval_workflow():
    cases = [
        EvalCase(
            id="Q1",
            question="What is A?",
            category="direct_factual",
            expected_answerable=True,
            reference_answer="Ans A",
            expected_citations=[ExpectedCitation(filename="a.pdf", page_number=1, chunk_id="cA")],
        ),
        EvalCase(
            id="Q2",
            question="What is B?",
            category="direct_factual",
            expected_answerable=True,
            reference_answer="Ans B",
            expected_citations=[ExpectedCitation(filename="b.pdf", page_number=1, chunk_id="cB")],
        ),
        EvalCase(
            id="Q3",
            question="What is unknown?",
            category="unanswerable",
            expected_answerable=False,
            reference_answer="I don't know",
            expected_citations=[],
        ),
    ]

    retriever = DummyRetriever({
        "What is A?": ["cA", "cX", "cY"],  # Rank 1 hit
        "What is B?": ["cX", "cB", "cY"],  # Rank 2 hit
        "What is unknown?": ["cX", "cY"],
    })

    report = evaluate_retrieval(cases, retriever=retriever, retriever_name="MockBM25", top_k=3)

    assert report.total_cases == 3
    assert report.answerable_cases == 2
    assert report.unanswerable_cases == 1

    # Q1: R@1 = 1.0, RR = 1.0, nDCG = 1.0
    # Q2: R@1 = 0.0, RR = 0.5, R@3 = 1.0
    # Mean R@1: (1.0 + 0.0) / 2 = 0.5
    assert report.mean_recall_at_1 == 0.5
    # MRR: (1.0 + 0.5) / 2 = 0.75
    assert report.mrr == 0.75
    assert report.mean_recall_at_3 == 1.0
    assert len(report.per_case_results) == 3


def test_ndcg_deduplication():
    gold = {"c1"}
    # If a buggy retriever returns duplicates, nDCG should not count c1 twice
    ndcg_duplicate = compute_ndcg_at_k(["c1", "c1", "c2"], gold, k=3)
    assert ndcg_duplicate == pytest.approx(1.0)


def test_evaluate_retrieval_answerable_without_gold_chunks_raises():
    cases = [
        EvalCase(
            id="Q_BAD",
            question="Bad?",
            category="direct_factual",
            expected_answerable=True,
            reference_answer="Bad",
            expected_citations=[],
        )
    ]
    retriever = DummyRetriever({})
    with pytest.raises(ValueError, match="has no valid gold chunk IDs"):
        evaluate_retrieval(cases, retriever)

