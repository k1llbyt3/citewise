"""Deterministic answer and citation evaluation engine for CiteWise RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any, Optional

from citewise.answer_generator import INSUFFICIENT_EVIDENCE
from citewise.eval_dataset import EvalCase


@dataclass
class CaseAnswerResult:
    case_id: str
    question: str
    category: str
    expected_answerable: bool
    model_answerable: bool
    correct_decision: bool
    correct_abstention: Optional[bool]
    actual_answer: str
    concept_coverage: float
    missing_concepts: list[str]
    citation_precision: float
    citation_recall: float
    citation_f1: float
    page_precision: float
    page_recall: float
    is_grounded_in_supplied: bool
    unsupplied_citations: list[dict[str, Any]]
    document_coverage: Optional[float] = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    is_error: bool = False
    error_message: Optional[str] = None


@dataclass
class AnswerEvaluationReport:
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    answerability_accuracy: float
    abstention_accuracy: float
    mean_concept_coverage: float
    mean_citation_precision: float
    mean_citation_recall: float
    mean_citation_f1: float
    mean_page_precision: float
    mean_page_recall: float
    grounded_citation_rate: float
    error_count: int
    mean_document_coverage: Optional[float] = None
    is_mock_generation: bool = False
    per_case_results: list[CaseAnswerResult] = field(default_factory=list)


def is_abstention(answer: Any, citations: Any) -> bool:
    """Determine whether a generation result represents an explicit abstention."""
    if not isinstance(citations, list) or not citations:
        return True
    if not isinstance(answer, str):
        return True
    normalized_answer = answer.strip().lower()
    insufficient_prefix = "i couldn't find enough information"
    if normalized_answer.startswith(insufficient_prefix):
        return True
    if INSUFFICIENT_EVIDENCE.lower() in normalized_answer:
        return True
    return False


def compute_concept_coverage(
    answer: str,
    required_concepts: list[str],
) -> tuple[float, list[str]]:
    """Deterministic check of required concept terms within the generated answer."""
    if not required_concepts:
        return 1.0, []

    if not isinstance(answer, str) or not answer.strip():
        return 0.0, list(required_concepts)

    lower_answer = answer.lower()
    missing = []
    for concept in required_concepts:
        if concept.lower() not in lower_answer:
            missing.append(concept)

    hits = len(required_concepts) - len(missing)
    coverage = hits / len(required_concepts)
    return coverage, missing


def compute_citation_metrics(
    cited: list[dict[str, Any]],
    gold_citations: list[Any],
    expected_answerable: bool,
) -> dict[str, float]:
    """Compute precision, recall, and F1 for chunk IDs and (filename, page_number) pairs."""
    if not isinstance(cited, list):
        cited = []

    cited_chunks = {
        c["chunk_id"] for c in cited if isinstance(c, dict) and c.get("chunk_id")
    }
    gold_chunks = {
        c.chunk_id for c in gold_citations if getattr(c, "chunk_id", None)
    }

    cited_pages = {
        (c["filename"], c["page_number"])
        for c in cited
        if isinstance(c, dict) and "filename" in c and "page_number" in c
    }
    gold_pages = {
        (c.filename, c.page_number)
        for c in gold_citations
        if hasattr(c, "filename") and hasattr(c, "page_number")
    }

    if not expected_answerable:
        # If unanswerable, producing citations is an error (precision=0, recall=0).
        # Correctly producing 0 citations is a perfect 1.0.
        if not cited_chunks and not cited_pages:
            return {
                "chunk_precision": 1.0,
                "chunk_recall": 1.0,
                "chunk_f1": 1.0,
                "page_precision": 1.0,
                "page_recall": 1.0,
                "page_f1": 1.0,
            }
        return {
            "chunk_precision": 0.0,
            "chunk_recall": 0.0,
            "chunk_f1": 0.0,
            "page_precision": 0.0,
            "page_recall": 0.0,
            "page_f1": 0.0,
        }

    # Answerable case
    chunk_prec = (
        len(cited_chunks & gold_chunks) / len(cited_chunks)
        if cited_chunks
        else 0.0
    )
    chunk_rec = (
        len(cited_chunks & gold_chunks) / len(gold_chunks)
        if gold_chunks
        else 0.0
    )
    chunk_f1 = (
        (2 * chunk_prec * chunk_rec) / (chunk_prec + chunk_rec)
        if (chunk_prec + chunk_rec) > 0
        else 0.0
    )

    page_prec = (
        len(cited_pages & gold_pages) / len(cited_pages)
        if cited_pages
        else 0.0
    )
    page_rec = (
        len(cited_pages & gold_pages) / len(gold_pages)
        if gold_pages
        else 0.0
    )
    page_f1 = (
        (2 * page_prec * page_rec) / (page_prec + page_rec)
        if (page_prec + page_rec) > 0
        else 0.0
    )

    return {
        "chunk_precision": chunk_prec,
        "chunk_recall": chunk_rec,
        "chunk_f1": chunk_f1,
        "page_precision": page_prec,
        "page_recall": page_rec,
        "page_f1": page_f1,
    }


def compute_document_coverage(
    citations: list[dict[str, Any]],
    gold_citations: list[Any],
    expected_answerable: bool,
    model_answerable: bool = True,
) -> Optional[float]:
    """Compute required-document coverage (covered required docs / total required docs).

    Returns None for unanswerable cases or cases without required gold documents.
    Returns 0.0 if the model abstains or encounters an error on an answerable case.
    """
    if not expected_answerable:
        return None

    gold_docs = {
        c.filename
        for c in gold_citations
        if hasattr(c, "filename") and c.filename
    } if gold_citations else set()

    if not gold_docs and gold_citations:
        gold_docs = {
            c["filename"]
            for c in gold_citations
            if isinstance(c, dict) and c.get("filename")
        }

    if not gold_docs:
        return None

    if not model_answerable:
        return 0.0

    cited_docs = {
        c["filename"]
        for c in citations
        if isinstance(c, dict) and c.get("filename")
    }

    covered = cited_docs & gold_docs
    return round(len(covered) / len(gold_docs), 4)


def evaluate_single_answer(
    case: EvalCase,
    generated_result: dict[str, Any],
    supplied_passages: list[dict[str, Any]],
) -> CaseAnswerResult:
    """Evaluate a single answer result deterministically against ground truth."""
    if not isinstance(generated_result, dict):
        generated_result = {
            "answer": "",
            "citations": [],
            "is_error": True,
            "error_message": f"Malformed result: expected dictionary, got {type(generated_result).__name__}",
        }

    raw_answer = generated_result.get("answer", "")
    answer = str(raw_answer) if raw_answer is not None else ""
    raw_citations = generated_result.get("citations", [])
    citations = raw_citations if isinstance(raw_citations, list) else []
    is_err = bool(generated_result.get("is_error", False))
    err_msg = generated_result.get("error_message")

    abstained = is_abstention(answer, citations)
    model_answerable = not abstained and not is_err

    correct_decision = model_answerable == case.expected_answerable
    correct_abstention = (
        abstained if not case.expected_answerable else None
    )

    # Concept coverage
    if case.expected_answerable:
        if abstained or is_err:
            coverage = 0.0
            missing = list(case.required_concepts)
        else:
            coverage, missing = compute_concept_coverage(
                answer, case.required_concepts
            )
    else:
        # If unanswerable, successful abstention yields 1.0 coverage
        coverage = 1.0 if abstained else 0.0
        missing = []

    # Citation metrics
    cit_metrics = compute_citation_metrics(
        citations, case.expected_citations, case.expected_answerable
    )

    # Groundedness in supplied context check
    supplied_chunk_ids = {
        p["chunk_id"] for p in supplied_passages if isinstance(p, dict) and "chunk_id" in p
    }
    unsupplied = [
        c
        for c in citations
        if isinstance(c, dict) and c.get("chunk_id") and c["chunk_id"] not in supplied_chunk_ids
    ]
    is_grounded = len(unsupplied) == 0

    chunk_to_source = {
        p["chunk_id"]: f"S{idx}"
        for idx, p in enumerate(supplied_passages, start=1)
        if isinstance(p, dict) and "chunk_id" in p
    }

    enriched_citations = []
    derived_source_ids = []
    for c in citations:
        if isinstance(c, dict):
            s_id = c.get("source_id") or chunk_to_source.get(c.get("chunk_id"), "")
            enriched_citations.append({
                "source_id": s_id,
                "filename": c.get("filename", ""),
                "page_number": c.get("page_number", 0),
                "chunk_id": c.get("chunk_id", ""),
            })
            if s_id and s_id not in derived_source_ids:
                derived_source_ids.append(s_id)
        elif isinstance(c, str):
            derived_source_ids.append(c)

    raw_source_ids = generated_result.get("source_ids")
    if isinstance(raw_source_ids, list):
        for s in raw_source_ids:
            if isinstance(s, str) and s not in derived_source_ids:
                derived_source_ids.append(s)

    doc_coverage = compute_document_coverage(
        citations=citations,
        gold_citations=case.expected_citations,
        expected_answerable=case.expected_answerable,
        model_answerable=model_answerable,
    )

    return CaseAnswerResult(
        case_id=case.id,
        question=case.question,
        category=case.category,
        expected_answerable=case.expected_answerable,
        model_answerable=model_answerable,
        correct_decision=correct_decision,
        correct_abstention=correct_abstention,
        actual_answer=answer,
        concept_coverage=round(coverage, 4),
        missing_concepts=missing,
        citation_precision=round(cit_metrics["chunk_precision"], 4),
        citation_recall=round(cit_metrics["chunk_recall"], 4),
        citation_f1=round(cit_metrics["chunk_f1"], 4),
        page_precision=round(cit_metrics["page_precision"], 4),
        page_recall=round(cit_metrics["page_recall"], 4),
        is_grounded_in_supplied=is_grounded,
        unsupplied_citations=unsupplied,
        document_coverage=doc_coverage,
        citations=enriched_citations,
        source_ids=derived_source_ids,
        is_error=is_err,
        error_message=err_msg,
    )


def evaluate_answers(
    cases: list[EvalCase],
    results_by_case_id: dict[str, dict[str, Any]],
    passages_by_case_id: dict[str, list[dict[str, Any]]],
    is_mock_generation: bool = False,
) -> AnswerEvaluationReport:
    """Evaluate full batch of generated answers against gold cases."""
    case_results: list[CaseAnswerResult] = []
    total = len(cases)
    ans_count = 0
    unans_count = 0
    correct_decisions = 0
    correct_abstentions = 0
    err_count = 0

    sum_coverage = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_f1 = 0.0
    sum_page_prec = 0.0
    sum_page_rec = 0.0
    grounded_count = 0

    for case in cases:
        gen_result = results_by_case_id.get(
            case.id,
            {
                "answer": "",
                "citations": [],
                "is_error": True,
                "error_message": "Missing result for case",
            },
        )
        passages = passages_by_case_id.get(case.id, [])
        res = evaluate_single_answer(case, gen_result, passages)
        case_results.append(res)

        if res.is_error:
            err_count += 1

        if case.expected_answerable:
            ans_count += 1
        else:
            unans_count += 1
            if res.correct_abstention:
                correct_abstentions += 1

        if res.correct_decision:
            correct_decisions += 1

        if res.is_grounded_in_supplied:
            grounded_count += 1

        sum_coverage += res.concept_coverage
        sum_prec += res.citation_precision
        sum_rec += res.citation_recall
        sum_f1 += res.citation_f1
        sum_page_prec += res.page_precision
        sum_page_rec += res.page_recall

    ans_acc = correct_decisions / total if total > 0 else 0.0
    abst_acc = correct_abstentions / unans_count if unans_count > 0 else 1.0

    doc_coverages = [
        r.document_coverage
        for r in case_results
        if r.document_coverage is not None
    ]
    mean_doc_cov = (
        round(sum(doc_coverages) / len(doc_coverages), 4)
        if doc_coverages
        else None
    )

    return AnswerEvaluationReport(
        total_cases=total,
        answerable_cases=ans_count,
        unanswerable_cases=unans_count,
        answerability_accuracy=round(ans_acc, 4),
        abstention_accuracy=round(abst_acc, 4),
        mean_concept_coverage=round(sum_coverage / total if total else 0.0, 4),
        mean_citation_precision=round(sum_prec / total if total else 0.0, 4),
        mean_citation_recall=round(sum_rec / total if total else 0.0, 4),
        mean_citation_f1=round(sum_f1 / total if total else 0.0, 4),
        mean_page_precision=round(sum_page_prec / total if total else 0.0, 4),
        mean_page_recall=round(sum_page_rec / total if total else 0.0, 4),
        grounded_citation_rate=round(
            grounded_count / total if total else 0.0, 4
        ),
        error_count=err_count,
        mean_document_coverage=mean_doc_cov,
        is_mock_generation=is_mock_generation,
        per_case_results=case_results,
    )
