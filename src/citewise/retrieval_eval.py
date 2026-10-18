"""Retrieval metrics and evaluation engine for CiteWise RAG pipeline."""

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from citewise.eval_dataset import EvalCase
from citewise.retriever import BM25Retriever


@dataclass
class CaseRetrievalResult:
    case_id: str
    question: str
    category: str
    expected_answerable: bool
    gold_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    ndcg_at_5: float
    first_hit_rank: Optional[int] = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalEvaluationReport:
    retriever_name: str
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mrr: float
    mean_ndcg_at_5: float
    per_case_results: list[CaseRetrievalResult] = field(default_factory=list)


def compute_recall_at_k(
    retrieved: list[str],
    gold: set[str],
    k: int,
) -> float:
    """Compute Recall@K: fraction of gold items found in top-K retrieved items."""
    if not gold:
        return 0.0
    top_k = retrieved[:k]
    hits = len(set(top_k) & gold)
    return hits / len(gold)


def compute_reciprocal_rank(
    retrieved: list[str],
    gold: set[str],
) -> tuple[float, Optional[int]]:
    """Compute Reciprocal Rank: 1 / rank of first relevant item (1-indexed)."""
    if not gold:
        return 0.0, None
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank, rank
    return 0.0, None


def compute_ndcg_at_k(
    retrieved: list[str],
    gold: set[str],
    k: int = 5,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """Compute normalized Discounted Cumulative Gain (nDCG@K)."""
    if not gold:
        return 0.0

    seen: set[str] = set()
    deduped = [item for item in retrieved if not (item in seen or seen.add(item))]
    top_k = deduped[:k]

    # Compute DCG@K
    dcg = 0.0
    for i, item in enumerate(top_k):
        rel = 0.0
        if relevance_scores and item in relevance_scores:
            rel = relevance_scores[item]
        elif item in gold:
            rel = 1.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)

    # Compute Ideal DCG@K
    if relevance_scores:
        ideal_rels = sorted(
            [relevance_scores.get(item, 1.0) for item in gold],
            reverse=True,
        )[:k]
    else:
        ideal_rels = [1.0] * min(len(gold), k)

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_retrieval(
    cases: list[EvalCase],
    retriever: Any,
    retriever_name: str = "BM25",
    top_k: int = 5,
) -> RetrievalEvaluationReport:
    """Evaluate a retriever instance against gold-standard evaluation cases.

    Only answerable cases with expected citations are factored into retrieval quality
    metrics (Recall, MRR, nDCG). Unanswerable queries are tracked separately.
    """
    results: list[CaseRetrievalResult] = []
    answerable_count = 0
    unanswerable_count = 0

    sum_r1 = 0.0
    sum_r3 = 0.0
    sum_r5 = 0.0
    sum_rr = 0.0
    sum_ndcg = 0.0

    for case in cases:
        retrieved_passages = retriever.search(case.question, top_k=top_k)
        retrieved_ids = [p["chunk_id"] for p in retrieved_passages]

        gold_chunk_ids = [
            c.chunk_id for c in case.expected_citations if c.chunk_id
        ]
        gold_set = set(gold_chunk_ids)

        if not case.expected_answerable:
            unanswerable_count += 1
            # For unanswerable queries, no relevance is expected
            results.append(
                CaseRetrievalResult(
                    case_id=case.id,
                    question=case.question,
                    category=case.category,
                    expected_answerable=case.expected_answerable,
                    gold_chunk_ids=[],
                    retrieved_chunk_ids=retrieved_ids,
                    recall_at_1=0.0,
                    recall_at_3=0.0,
                    recall_at_5=0.0,
                    reciprocal_rank=0.0,
                    ndcg_at_5=0.0,
                    first_hit_rank=None,
                    retrieved_chunks=retrieved_passages,
                )
            )
            continue

        if not gold_set:
            raise ValueError(
                f"Case {case.id} is marked answerable but has no valid gold chunk IDs."
            )

        answerable_count += 1
        r1 = compute_recall_at_k(retrieved_ids, gold_set, k=1)
        r3 = compute_recall_at_k(retrieved_ids, gold_set, k=3)
        r5 = compute_recall_at_k(retrieved_ids, gold_set, k=5)
        rr, hit_rank = compute_reciprocal_rank(retrieved_ids, gold_set)
        ndcg = compute_ndcg_at_k(retrieved_ids, gold_set, k=5)

        sum_r1 += r1
        sum_r3 += r3
        sum_r5 += r5
        sum_rr += rr
        sum_ndcg += ndcg

        results.append(
            CaseRetrievalResult(
                case_id=case.id,
                question=case.question,
                category=case.category,
                expected_answerable=case.expected_answerable,
                gold_chunk_ids=gold_chunk_ids,
                retrieved_chunk_ids=retrieved_ids,
                recall_at_1=r1,
                recall_at_3=r3,
                recall_at_5=r5,
                reciprocal_rank=rr,
                ndcg_at_5=ndcg,
                first_hit_rank=hit_rank,
                retrieved_chunks=retrieved_passages,
            )
        )

    mean_r1 = sum_r1 / answerable_count if answerable_count > 0 else 0.0
    mean_r3 = sum_r3 / answerable_count if answerable_count > 0 else 0.0
    mean_r5 = sum_r5 / answerable_count if answerable_count > 0 else 0.0
    mrr = sum_rr / answerable_count if answerable_count > 0 else 0.0
    mean_ndcg = sum_ndcg / answerable_count if answerable_count > 0 else 0.0

    return RetrievalEvaluationReport(
        retriever_name=retriever_name,
        total_cases=len(cases),
        answerable_cases=answerable_count,
        unanswerable_cases=unanswerable_count,
        mean_recall_at_1=round(mean_r1, 4),
        mean_recall_at_3=round(mean_r3, 4),
        mean_recall_at_5=round(mean_r5, 4),
        mrr=round(mrr, 4),
        mean_ndcg_at_5=round(mean_ndcg, 4),
        per_case_results=results,
    )


def main() -> int:
    import argparse
    from pathlib import Path
    from citewise.eval_dataset import validate_eval_dataset
    from citewise.retriever import load_chunks

    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality across gold-standard benchmark cases."
    )
    parser.add_argument(
        "--reports",
        default="evaluation/reports",
        help="Directory containing JSON ingestion reports.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.json",
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K passages to retrieve per query (default: 5).",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    if not reports_dir.is_dir():
        parser.error(f"Reports directory not found: {reports_dir}")

    report_files = sorted(reports_dir.glob("*.json"))
    if not report_files:
        parser.error(f"No JSON reports found in: {reports_dir}")

    chunks = load_chunks(report_files)
    cases = validate_eval_dataset(args.dataset, reports_dir=reports_dir)

    retriever = BM25Retriever(chunks)
    report = evaluate_retrieval(cases, retriever=retriever, retriever_name="BM25", top_k=args.top_k)

    print("\n================== BM25 RETRIEVAL EVALUATION REPORT ==================")
    print(f"Retriever:             {report.retriever_name}")
    print(f"Corpus Chunks:         {len(chunks)} across {len(report_files)} documents")
    print(f"Total Cases:           {report.total_cases}")
    print(f"Answerable Cases:      {report.answerable_cases}")
    print(f"Unanswerable Cases:    {report.unanswerable_cases}")
    print(f"Mean Recall@1:         {report.mean_recall_at_1:.4f}")
    print(f"Mean Recall@3:         {report.mean_recall_at_3:.4f}")
    print(f"Mean Recall@5:         {report.mean_recall_at_5:.4f}")
    print(f"MRR (Mean Recip Rank): {report.mrr:.4f}")
    print(f"Mean nDCG@5:           {report.mean_ndcg_at_5:.4f}")
    print("======================================================================\n")

    print("Per-Case Results:")
    for r in report.per_case_results:
        if r.expected_answerable:
            hit_str = f"Rank {r.first_hit_rank}" if r.first_hit_rank else "MISS"
            print(f"  [{r.case_id}] {r.category:<22} | {hit_str:<7} | R@1={r.recall_at_1:.2f} R@5={r.recall_at_5:.2f} nDCG@5={r.ndcg_at_5:.3f} | {r.question[:50]}...")
        else:
            print(f"  [{r.case_id}] {r.category:<22} | UNANSWERABLE (Correctly expected 0 hits)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

