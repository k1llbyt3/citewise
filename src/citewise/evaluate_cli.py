"""Professional RAG evaluation CLI for CiteWise."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from citewise.answer_eval import evaluate_answers
from citewise.answer_generator import generate_answer
from citewise.api_limits import wait_for_api_slot
from citewise.eval_dataset import EvalCase, validate_eval_dataset
from citewise.query_cache import (
    DEFAULT_QUERY_CACHE_PATH,
    QueryCacheMissError,
    QueryEmbeddingCache,
)
from citewise.retrieval_eval import evaluate_retrieval
from citewise.retriever import BM25Retriever, load_chunks


DEFAULT_REPORTS_DIR = "evaluation/reports"
DEFAULT_DATASET_PATH = "evaluation/dataset.json"
DEFAULT_INDEX_PATH = "evaluation/data/semantic_index.json"
DEFAULT_RESULTS_DIR = "evaluation/results"
MAX_LIVE_CASES = 3


def compute_file_hash(path: Path) -> str:
    """Compute sha256 hash of a file for run provenance."""
    if not path.is_file():
        return "not_found"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CiteWise Professional RAG Evaluation Framework CLI."
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "answer", "full"],
        default="retrieval",
        help="Evaluation mode: retrieval only, answer only, or full pipeline (default: retrieval).",
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "semantic", "hybrid"],
        default="bm25",
        help="Retriever algorithm to evaluate (default: bm25).",
    )
    parser.add_argument(
        "--reports",
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory containing JSON ingestion reports (default: {DEFAULT_REPORTS_DIR}).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to evaluation dataset JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Path to semantic index (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory where evaluation results are written (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Opt-in to live API calls. When omitted, runs completely offline.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=f"Limit the number of evaluation cases. In live mode, cannot exceed {MAX_LIVE_CASES}.",
    )
    parser.add_argument(
        "--query-cache",
        default=DEFAULT_QUERY_CACHE_PATH,
        help=f"Path to query embedding cache JSON (default: {DEFAULT_QUERY_CACHE_PATH}).",
    )
    parser.add_argument(
        "--allow-query-embedding-api",
        action="store_true",
        help="Opt-in to calling Gemini API to embed queries and populate the query cache. When omitted, retrieval runs completely offline against the cache.",
    )
    parser.add_argument(
        "--mock-responses",
        default=None,
        help="Optional path to JSON file containing predetermined mock generation responses for offline testing.",
    )
    return parser


def run_offline_mock_generation(
    question: str,
    passages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic offline answer generator for offline testing without API credits."""
    from citewise.answer_generator import INSUFFICIENT_EVIDENCE

    if not passages:
        return {"answer": INSUFFICIENT_EVIDENCE, "citations": []}

    # Simulate grounded generation using highest-ranked passage
    top_p = passages[0]
    return {
        "answer": f"Based on the provided PDF passage, {top_p['text'][:150]}...",
        "citations": [
            {
                "filename": top_p["filename"],
                "page_number": top_p["page_number"],
                "chunk_id": top_p["chunk_id"],
            }
        ],
    }


def write_evaluation_reports(
    results_dir: Path,
    run_id: str,
    run_metadata: dict[str, Any],
    retrieval_report: Optional[Any],
    answer_report: Optional[Any],
) -> tuple[Path, Path]:
    """Write machine-readable JSON and human-readable Markdown evaluation reports."""
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"{run_id}.json"
    md_path = results_dir / f"{run_id}.md"

    # Prepare structured JSON payload
    payload: dict[str, Any] = {
        "metadata": run_metadata,
        "retrieval": None,
        "answer": None,
    }

    if retrieval_report:
        payload["retrieval"] = {
            "retriever": retrieval_report.retriever_name,
            "total_cases": retrieval_report.total_cases,
            "answerable_cases": retrieval_report.answerable_cases,
            "unanswerable_cases": retrieval_report.unanswerable_cases,
            "mean_recall_at_1": retrieval_report.mean_recall_at_1,
            "mean_recall_at_3": retrieval_report.mean_recall_at_3,
            "mean_recall_at_5": retrieval_report.mean_recall_at_5,
            "mrr": retrieval_report.mrr,
            "mean_ndcg_at_5": retrieval_report.mean_ndcg_at_5,
            "per_case": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "expected_answerable": r.expected_answerable,
                    "gold_chunk_ids": r.gold_chunk_ids,
                    "retrieved_chunk_ids": r.retrieved_chunk_ids,
                    "first_hit_rank": r.first_hit_rank,
                    "recall_at_1": r.recall_at_1,
                    "recall_at_5": r.recall_at_5,
                    "reciprocal_rank": r.reciprocal_rank,
                    "ndcg_at_5": r.ndcg_at_5,
                }
                for r in retrieval_report.per_case_results
            ],
        }

    if answer_report:
        payload["answer"] = {
            "total_cases": answer_report.total_cases,
            "is_mock_generation": answer_report.is_mock_generation,
            "simulation_notice": (
                "Offline simulation run. Metrics evaluate scoring harness functionality, NOT real LLM answer quality."
                if answer_report.is_mock_generation
                else None
            ),
            "answerable_cases": answer_report.answerable_cases,
            "unanswerable_cases": answer_report.unanswerable_cases,
            "answerability_accuracy": answer_report.answerability_accuracy,
            "abstention_accuracy": answer_report.abstention_accuracy,
            "mean_concept_coverage": answer_report.mean_concept_coverage,
            "mean_document_coverage": getattr(answer_report, "mean_document_coverage", None),
            "mean_citation_precision": answer_report.mean_citation_precision,
            "mean_citation_recall": answer_report.mean_citation_recall,
            "mean_citation_f1": answer_report.mean_citation_f1,
            "mean_page_precision": answer_report.mean_page_precision,
            "mean_page_recall": answer_report.mean_page_recall,
            "grounded_citation_rate": answer_report.grounded_citation_rate,
            "error_count": answer_report.error_count,
            "per_case": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "question": r.question,
                    "expected_answerable": r.expected_answerable,
                    "model_answerable": r.model_answerable,
                    "correct_decision": r.correct_decision,
                    "actual_answer": r.actual_answer,
                    "source_ids": getattr(r, "source_ids", []),
                    "citations": getattr(r, "citations", []),
                    "concept_coverage": r.concept_coverage,
                    "missing_concepts": r.missing_concepts,
                    "document_coverage": getattr(r, "document_coverage", None),
                    "citation_precision": r.citation_precision,
                    "citation_recall": r.citation_recall,
                    "citation_f1": r.citation_f1,
                    "page_precision": r.page_precision,
                    "page_recall": r.page_recall,
                    "is_grounded_in_supplied": r.is_grounded_in_supplied,
                    "is_error": r.is_error,
                    "error_message": r.error_message,
                }
                for r in answer_report.per_case_results
            ],
        }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Generate Markdown report
    md_lines = [
        f"# CiteWise RAG Evaluation Report: `{run_id}`",
        "",
        f"- **Timestamp**: {run_metadata['timestamp']}",
        f"- **Mode**: `{run_metadata['mode']}`",
        f"- **Retriever**: `{run_metadata['retriever']}`",
        f"- **Live API Execution**: `{run_metadata['live_mode']}`",
        f"- **Generator Type**: `{run_metadata.get('generator_type', 'unknown')}`",
        f"- **Mock Generation**: `{run_metadata.get('is_mock_generation', False)}`",
        f"- **Allow Query Embedding API**: `{run_metadata.get('allow_query_embedding_api', False)}`",
        f"- **Query Cache**: `{run_metadata.get('query_cache_path', 'none')}`",
        f"- **Dataset**: `{run_metadata['dataset_path']}` (hash: `{run_metadata['dataset_hash']}`)",
        f"- **Cases Evaluated**: {run_metadata['cases_evaluated']}",
        "",
    ]

    if retrieval_report:
        md_lines.extend(
            [
                "## Retrieval Metrics",
                "",
                "| Metric | Score |",
                "| :--- | :---: |",
                f"| **Mean Recall@1** | {retrieval_report.mean_recall_at_1:.4f} |",
                f"| **Mean Recall@3** | {retrieval_report.mean_recall_at_3:.4f} |",
                f"| **Mean Recall@5** | {retrieval_report.mean_recall_at_5:.4f} |",
                f"| **Mean Reciprocal Rank (MRR)** | {retrieval_report.mrr:.4f} |",
                f"| **Mean nDCG@5** | {retrieval_report.mean_ndcg_at_5:.4f} |",
                "",
            ]
        )

    if answer_report:
        if answer_report.is_mock_generation:
            md_lines.extend(
                [
                    "> [!WARNING]",
                    "> **Offline Simulation Notice**: This evaluation was executed without `--live`.",
                    "> Answers were generated by an offline mock harness (`run_offline_mock_generation` or test fixture).",
                    "> These metrics evaluate scoring harness functionality only and DO NOT represent real Gemini LLM answer quality or actual model accuracy.",
                    "",
                ]
            )
        doc_cov_display = (
            f"{answer_report.mean_document_coverage:.4f}"
            if getattr(answer_report, "mean_document_coverage", None) is not None
            else "N/A"
        )
        md_lines.extend(
            [
                "## Answer & Citation Metrics",
                "",
                "| Metric | Score |",
                "| :--- | :---: |",
                f"| **Answerability Accuracy** | {answer_report.answerability_accuracy:.4f} |",
                f"| **Abstention Accuracy** | {answer_report.abstention_accuracy:.4f} |",
                f"| **Mean Concept Coverage** | {answer_report.mean_concept_coverage:.4f} |",
                f"| **Mean Document Coverage** | {doc_cov_display} |",
                f"| **Citation Precision (Chunk)** | {answer_report.mean_citation_precision:.4f} |",
                f"| **Citation Recall (Chunk)** | {answer_report.mean_citation_recall:.4f} |",
                f"| **Citation F1 (Chunk)** | {answer_report.mean_citation_f1:.4f} |",
                f"| **Citation Precision (Page)** | {answer_report.mean_page_precision:.4f} |",
                f"| **Citation Recall (Page)** | {answer_report.mean_page_recall:.4f} |",
                f"| **Grounded Citation Rate** | {answer_report.grounded_citation_rate:.4f} |",
                f"| **Execution Errors** | {answer_report.error_count} |",
                "",
            ]
        )

    md_lines.extend(
        [
            "## Per-Case Breakdown",
            "",
            "| Case ID | Category | Expected Ans | Status | Notes |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ]
    )

    all_cases = []
    if retrieval_report:
        for r in retrieval_report.per_case_results:
            hit = f"Rank {r.first_hit_rank}" if r.first_hit_rank else "Miss"
            ans_str = "Yes" if r.expected_answerable else "No"
            all_cases.append(
                f"| `{r.case_id}` | `{r.category}` | {ans_str} | {hit} | R@5={r.recall_at_5:.2f}, nDCG={r.ndcg_at_5:.2f} |"
            )
    elif answer_report:
        for r in answer_report.per_case_results:
            dec = "Correct" if r.correct_decision else "Incorrect"
            ans_str = "Yes" if r.expected_answerable else "No"
            all_cases.append(
                f"| `{r.case_id}` | `{r.category}` | {ans_str} | {dec} | Coverage={r.concept_coverage:.2f}, Prec={r.citation_precision:.2f} |"
            )

    md_lines.extend(all_cases)
    md_lines.append("")

    if answer_report:
        md_lines.extend(
            [
                "## Detailed Case Inspection",
                "",
            ]
        )
        for r in answer_report.per_case_results:
            exp_label = "Answerable" if r.expected_answerable else "Unanswerable / Abstain"
            model_label = "Answerable" if r.model_answerable else "Abstained"
            status_label = "Correct" if r.correct_decision else "INCORRECT"

            md_lines.extend(
                [
                    f"### `{r.case_id}` ({r.category})",
                    f"- **Question**: {r.question}",
                    f"- **Decision**: Expected `{exp_label}`, Model `{model_label}` (**{status_label}**)",
                    "- **Generated Answer**:",
                    f"  > {r.actual_answer.strip() if r.actual_answer.strip() else '*(empty)*'}",
                ]
            )

            citations = getattr(r, "citations", [])
            if citations:
                cit_items = [
                    f"`[{c.get('source_id') or 'Src'}]` {c.get('filename')}, page {c.get('page_number')} (chunk `{c.get('chunk_id')}`)"
                    for c in citations
                ]
                md_lines.append(f"- **Citations**: {', '.join(cit_items)}")
            else:
                md_lines.append("- **Citations**: None (Abstained)")

            doc_cov_val = (
                f"{r.document_coverage:.2f}"
                if getattr(r, "document_coverage", None) is not None
                else "N/A"
            )
            md_lines.append(
                f"- **Metrics**: Coverage={r.concept_coverage:.2f}, DocCoverage={doc_cov_val}, Precision={r.citation_precision:.2f}, Recall={r.citation_recall:.2f}, Grounded={r.is_grounded_in_supplied}"
            )
            if r.missing_concepts:
                md_lines.append(f"- **Missing Concepts**: {', '.join(r.missing_concepts)}")
            if r.is_error:
                md_lines.append(f"- **Error**: {r.error_message}")
            md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reports_dir = Path(args.reports)
    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)

    # 1. Validation & Pre-checks
    if not reports_dir.is_dir():
        print(f"Error: Reports directory not found: {reports_dir}", file=sys.stderr)
        return 1

    if not dataset_path.is_file():
        print(f"Error: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    # Guardrails for Live API Mode
    if args.live:
        if args.max_cases is None or args.max_cases > MAX_LIVE_CASES:
            print(
                f"Error: Live evaluation is strictly capped at {MAX_LIVE_CASES} cases. "
                f"Pass --max-cases <= {MAX_LIVE_CASES} to execute.",
                file=sys.stderr,
            )
            return 1

    if args.retriever in {"semantic", "hybrid"}:
        index_path = Path(args.index)
        if not index_path.is_file():
            print(
                f"Error: Semantic index '{index_path}' not found. "
                "Build the index before evaluating semantic/hybrid retrieval.",
                file=sys.stderr,
            )
            return 1

    try:
        report_files = sorted(reports_dir.glob("*.json"))
        if not report_files:
            raise ValueError(f"No JSON reports found in: {reports_dir}")

        chunks = load_chunks(report_files)
        all_cases = validate_eval_dataset(dataset_path, reports_dir=reports_dir)

        # Apply max-cases limit if specified
        cases = all_cases[: args.max_cases] if args.max_cases else all_cases

        if args.live and len(cases) > MAX_LIVE_CASES:
            print(
                f"Error: Cannot run {len(cases)} cases in live mode. Maximum is {MAX_LIVE_CASES}.",
                file=sys.stderr,
            )
            return 1

        # Instantiate requested retriever
        if args.retriever == "bm25":
            retriever = BM25Retriever(chunks)
        elif args.retriever in {"semantic", "hybrid"}:
            from citewise.embedding_index import load_embedding_index

            index = load_embedding_index(args.index)
            query_cache = QueryEmbeddingCache(
                cache_path=args.query_cache,
                allow_api=args.allow_query_embedding_api,
            )
            if args.retriever == "semantic":
                from citewise.semantic_retriever import SemanticRetriever

                retriever = SemanticRetriever(index, query_cache=query_cache)
            else:
                from citewise.hybrid_retriever import HybridRetriever

                retriever = HybridRetriever(
                    chunks=chunks, index=index, query_cache=query_cache
                )

    except Exception as exc:
        print(f"Initialization Error: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"eval_{timestamp}_{args.mode}_{args.retriever}"

    is_mock = not args.live
    gen_type = (
        "live_llm"
        if args.live
        else ("fixture_mock" if args.mock_responses else "offline_simulation_mock")
    )

    run_metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "mode": args.mode,
        "retriever": args.retriever,
        "live_mode": args.live,
        "generator_type": gen_type,
        "is_mock_generation": is_mock,
        "mock_responses_path": str(args.mock_responses) if args.mock_responses else None,
        "allow_query_embedding_api": args.allow_query_embedding_api,
        "query_cache_path": str(args.query_cache) if args.retriever in {"semantic", "hybrid"} else None,
        "top_k": args.top_k,
        "dataset_path": str(dataset_path),
        "dataset_hash": compute_file_hash(dataset_path),
        "reports_path": str(reports_dir),
        "cases_evaluated": len(cases),
    }

    retrieval_report = None
    answer_report = None

    # 2. Execute Evaluation Modes
    try:
        if args.mode in {"retrieval", "full"}:
            print(f"Running retrieval evaluation using '{args.retriever}' across {len(cases)} cases...")
            retrieval_report = evaluate_retrieval(
                cases,
                retriever=retriever,
                retriever_name=args.retriever.upper(),
                top_k=args.top_k,
            )

        if args.mode in {"answer", "full"}:
            gen_label = "live API" if args.live else ("fixture mock" if args.mock_responses else "offline mock simulator")
            print(f"Running answer evaluation across {len(cases)} cases (mode={gen_label})...")
            if is_mock:
                print(
                    "[NOTICE] Running in OFFLINE mock evaluation mode. "
                    "Metrics reflect test harness/mock responses, NOT real Gemini LLM answer quality."
                )

            mock_fixture_data = {}
            if args.mock_responses:
                mock_path = Path(args.mock_responses)
                if not mock_path.is_file():
                    print(f"Error: Mock responses file not found: {mock_path}", file=sys.stderr)
                    return 1
                raw_mock = json.loads(mock_path.read_text(encoding="utf-8"))
                mock_fixture_data = raw_mock.get("responses", raw_mock) if isinstance(raw_mock, dict) else {}

            results_by_case = {}
            passages_by_case = {}

            for case in cases:
                passages = retriever.search(case.question, top_k=args.top_k)
                passages_by_case[case.id] = passages

                if args.live:
                    wait_for_api_slot()
                    try:
                        res = generate_answer(case.question, passages)
                        results_by_case[case.id] = res
                    except Exception as exc:
                        results_by_case[case.id] = {
                            "answer": "",
                            "citations": [],
                            "is_error": True,
                            "error_message": str(exc),
                        }
                elif case.id in mock_fixture_data:
                    results_by_case[case.id] = mock_fixture_data[case.id]
                else:
                    # Deterministic offline mock generator
                    results_by_case[case.id] = run_offline_mock_generation(
                        case.question, passages
                    )

            answer_report = evaluate_answers(
                cases,
                results_by_case_id=results_by_case,
                passages_by_case_id=passages_by_case,
                is_mock_generation=is_mock,
            )
    except QueryCacheMissError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 3. Output Reports
    json_path, md_path = write_evaluation_reports(
        results_dir=results_dir,
        run_id=run_id,
        run_metadata=run_metadata,
        retrieval_report=retrieval_report,
        answer_report=answer_report,
    )

    print("\n================ EVALUATION COMPLETED ================")
    print(f"Run ID:        {run_id}")
    print(f"Mode:          {args.mode}")
    print(f"Retriever:     {args.retriever}")
    print(f"Cases:         {len(cases)}")
    if retrieval_report:
        print(f"Mean Recall@5: {retrieval_report.mean_recall_at_5:.4f}")
        print(f"MRR:           {retrieval_report.mrr:.4f}")
        print(f"Mean nDCG@5:   {retrieval_report.mean_ndcg_at_5:.4f}")
    if answer_report:
        print(f"Answerability: {answer_report.answerability_accuracy:.4f}")
        print(f"Abstention:    {answer_report.abstention_accuracy:.4f}")
        print(f"Concept Cov:   {answer_report.mean_concept_coverage:.4f}")
        if getattr(answer_report, "mean_document_coverage", None) is not None:
            print(f"Doc Coverage:  {answer_report.mean_document_coverage:.4f}")
    print(f"JSON Report:   {json_path}")
    print(f"Markdown:      {md_path}")
    print("======================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
