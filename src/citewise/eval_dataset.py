"""Schema definition and validation for CiteWise gold-standard RAG evaluation dataset."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_CATEGORIES = {
    "direct_factual",
    "paraphrased_semantic",
    "multi_passage",
    "multi_page",
    "cross_document",
    "ocr_scanned",
    "semantic_vs_bm25",
    "unanswerable",
    "misleading_premise",
    "ambiguous_insufficient",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class ExpectedCitation:
    filename: str
    page_number: int
    chunk_id: Optional[str] = None


@dataclass
class EvalCase:
    id: str
    question: str
    category: str
    expected_answerable: bool
    reference_answer: str
    required_concepts: list[str] = field(default_factory=list)
    expected_citations: list[ExpectedCitation] = field(default_factory=list)
    difficulty: str = "medium"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "expected_answerable": self.expected_answerable,
            "reference_answer": self.reference_answer,
            "required_concepts": self.required_concepts,
            "expected_citations": [
                {
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_id": c.chunk_id,
                }
                for c in self.expected_citations
            ],
            "difficulty": self.difficulty,
            "rationale": self.rationale,
        }


def validate_eval_dataset(
    dataset_path: str | Path,
    reports_dir: Optional[str | Path] = None,
) -> list[EvalCase]:
    """Load and validate an evaluation dataset against the active reports corpus."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in dataset {path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise ValueError("Evaluation dataset must be a JSON array of cases.")

    if not raw_data:
        raise ValueError("Evaluation dataset cannot be empty.")

    # Optionally load valid chunk IDs and document filenames from reports
    valid_chunks_by_file: dict[str, dict[int, set[str]]] = {}
    valid_files: set[str] = set()

    if reports_dir:
        r_dir = Path(reports_dir)
        if not r_dir.is_dir():
            raise FileNotFoundError(f"Reports directory not found: {r_dir}")
        for report_file in r_dir.glob("*.json"):
            rep_data = json.loads(report_file.read_text(encoding="utf-8"))
            fname = rep_data["document"]["filename"]
            valid_files.add(fname)
            if fname not in valid_chunks_by_file:
                valid_chunks_by_file[fname] = {}
            for chunk in rep_data.get("chunks", []):
                p_num = chunk["page_number"]
                if p_num not in valid_chunks_by_file[fname]:
                    valid_chunks_by_file[fname][p_num] = set()
                valid_chunks_by_file[fname][p_num].add(chunk["chunk_id"])

    seen_ids = set()
    validated_cases: list[EvalCase] = []

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{idx} must be a JSON object.")

        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case #{idx} missing valid 'id'.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id '{case_id}' in dataset.")
        seen_ids.add(case_id)

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {case_id} missing valid 'question'.")

        category = item.get("category")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Case {case_id} has invalid category '{category}'. "
                f"Must be one of {sorted(VALID_CATEGORIES)}."
            )

        expected_answerable = item.get("expected_answerable")
        if not isinstance(expected_answerable, bool):
            raise ValueError(
                f"Case {case_id} 'expected_answerable' must be a boolean."
            )

        ref_answer = item.get("reference_answer", "")
        if not isinstance(ref_answer, str):
            raise ValueError(
                f"Case {case_id} 'reference_answer' must be a string."
            )

        if expected_answerable and not ref_answer.strip():
            raise ValueError(
                f"Case {case_id} is marked answerable but has an empty reference_answer."
            )

        req_concepts = item.get("required_concepts", [])
        if not isinstance(req_concepts, list) or any(
            not isinstance(c, str) for c in req_concepts
        ):
            raise ValueError(
                f"Case {case_id} 'required_concepts' must be a list of strings."
            )

        diff = item.get("difficulty", "medium")
        if diff not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Case {case_id} has invalid difficulty '{diff}'. "
                f"Must be one of {sorted(VALID_DIFFICULTIES)}."
            )

        raw_cits = item.get("expected_citations", [])
        if not isinstance(raw_cits, list):
            raise ValueError(
                f"Case {case_id} 'expected_citations' must be a list."
            )

        citations: list[ExpectedCitation] = []
        for c_idx, cit in enumerate(raw_cits):
            if not isinstance(cit, dict):
                raise ValueError(
                    f"Case {case_id} citation #{c_idx} must be a dictionary."
                )
            fname = cit.get("filename")
            pnum = cit.get("page_number")
            cid = cit.get("chunk_id")

            if not isinstance(fname, str) or not fname.strip():
                raise ValueError(
                    f"Case {case_id} citation #{c_idx} missing valid 'filename'."
                )
            if not isinstance(pnum, int) or isinstance(pnum, bool) or pnum < 1:
                raise ValueError(
                    f"Case {case_id} citation #{c_idx} page_number must be a positive integer."
                )
            if cid is not None and (not isinstance(cid, str) or not cid.strip()):
                raise ValueError(
                    f"Case {case_id} citation #{c_idx} chunk_id must be a string or null."
                )

            # Validate against actual corpus reports if provided
            if reports_dir:
                if fname not in valid_files:
                    raise ValueError(
                        f"Case {case_id} cites non-existent file '{fname}'. Active files: {valid_files}"
                    )
                if pnum not in valid_chunks_by_file[fname]:
                    raise ValueError(
                        f"Case {case_id} cites page {pnum} which does not exist in {fname}."
                    )
                if cid and cid not in valid_chunks_by_file[fname][pnum]:
                    raise ValueError(
                        f"Case {case_id} cites chunk_id '{cid}' which does not exist on page {pnum} of {fname}."
                    )

            citations.append(
                ExpectedCitation(
                    filename=fname,
                    page_number=pnum,
                    chunk_id=cid,
                )
            )

        if expected_answerable and not citations:
            raise ValueError(
                f"Case {case_id} is marked answerable but provides no expected citations."
            )

        if expected_answerable and not any(c.chunk_id for c in citations):
            raise ValueError(
                f"Case {case_id} is marked answerable but provides no citation with a valid chunk_id."
            )

        if not expected_answerable and citations:
            raise ValueError(
                f"Case {case_id} is unanswerable but provides expected citations."
            )

        rationale = item.get("rationale", "")

        validated_cases.append(
            EvalCase(
                id=case_id,
                question=question,
                category=category,
                expected_answerable=expected_answerable,
                reference_answer=ref_answer,
                required_concepts=req_concepts,
                expected_citations=citations,
                difficulty=diff,
                rationale=rationale,
            )
        )

    return validated_cases
