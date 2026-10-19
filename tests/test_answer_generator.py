
import json

import pytest

from citewise.answer_generator import (
    INSUFFICIENT_EVIDENCE,
    generate_answer,
)


class FakeResponse:
    def __init__(self, payload):
        self.text = json.dumps(payload)


class FakeModels:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        if self.error:
            raise self.error

        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.models = FakeModels(payload=payload, error=error)


def make_passages():
    return [
        {
            "chunk_id": "chunk-1",
            "filename": "text_sample.pdf",
            "page_number": 2,
            "text": (
                "Chunking splits long documents into smaller passages. "
                "Each chunk retains its source page."
            ),
        }
    ]


def test_answer_uses_retrieved_evidence_and_real_citation():
    client = FakeClient({
        "answerable": True,
        "answer": (
            "Chunking divides long documents into smaller passages, "
            "and each chunk retains its source page."
        ),
        "source_ids": ["S1"],
    })

    result = generate_answer(
        "Why is chunking useful?",
        make_passages(),
        client=client,
    )

    assert "smaller passages" in result["answer"]
    assert result["citations"] == [
        {
            "filename": "text_sample.pdf",
            "page_number": 2,
            "chunk_id": "chunk-1",
        }
    ]


def test_prompt_forbids_outside_knowledge():
    client = FakeClient({
        "answerable": True,
        "answer": "Chunking creates smaller passages.",
        "source_ids": ["S1"],
    })

    generate_answer(
        "What does chunking do?",
        make_passages(),
        client=client,
    )

    prompt = client.models.calls[0]["contents"]

    assert "ONLY the supplied PDF passages" in prompt
    assert "Do NOT use general knowledge" in prompt
    assert "Do NOT add facts" in prompt
    assert "text_sample.pdf" in prompt


def test_insufficient_evidence_returns_fixed_fallback():
    client = FakeClient({
        "answerable": False,
        "answer": "The documents do not explain this.",
        "source_ids": [],
    })

    result = generate_answer(
        "What is the capital of France?",
        make_passages(),
        client=client,
    )

    assert result == {
        "answer": INSUFFICIENT_EVIDENCE,
        "citations": [],
    }


def test_empty_passages_do_not_call_gemini():
    client = FakeClient({
        "answerable": True,
        "answer": "Unexpected answer",
        "source_ids": ["S1"],
    })

    result = generate_answer(
        "Any question",
        [],
        client=client,
    )

    assert result["answer"] == INSUFFICIENT_EVIDENCE
    assert result["citations"] == []
    assert client.models.calls == []


def test_unknown_citation_is_rejected():
    client = FakeClient({
        "answerable": True,
        "answer": "An unsupported answer.",
        "source_ids": ["S99"],
    })

    with pytest.raises(ValueError, match="citation that was not supplied"):
        generate_answer(
            "What does the PDF say?",
            make_passages(),
            client=client,
        )


def test_gemini_api_failure_is_propagated():
    client = FakeClient(error=RuntimeError("Simulated API failure"))

    with pytest.raises(RuntimeError, match="Simulated API failure"):
        generate_answer(
            "Why is chunking useful?",
            make_passages(),
            client=client,
        )


def test_invalid_json_response_is_rejected():
    class InvalidModels:
        def generate_content(self, **kwargs):
            class Response:
                text = "This is not JSON"

            return Response()

    class InvalidClient:
        models = InvalidModels()

    with pytest.raises(ValueError, match="invalid JSON"):
        generate_answer(
            "Why is chunking useful?",
            make_passages(),
            client=InvalidClient(),
        )


def test_prompt_demands_direct_responsiveness_and_definition_priority():
    """Verify system prompt mandates concise direct answer and prioritization of direct definitions."""
    client = FakeClient({
        "answerable": True,
        "answer": "A variable is a named memory location.",
        "source_ids": ["S1"],
    })

    generate_answer(
        "what are variables?",
        make_passages(),
        client=client,
    )

    prompt = client.models.calls[0]["contents"]
    assert "Provide a concise, direct answer specifically responsive" in prompt
    assert "Prioritize passages that directly define, state, or explain" in prompt
    assert "Every factual claim must be explicitly supported by the cited passages" in prompt


def test_generate_answer_prioritizes_definition_passage():
    """Verify that when direct definition and tangential passages are given, definition is cited."""
    passages = [
        {
            "chunk_id": "c-def",
            "filename": "WebTechnology_Module 2 NOTES.pdf",
            "page_number": 29,
            "text": "Variables:\nA variable is named memory location to store data. JavaScript is loosely typed.",
        },
        {
            "chunk_id": "c-tangent",
            "filename": "WebTechnology_Module 2 NOTES.pdf",
            "page_number": 30,
            "text": "typeof operator accepts a single operand either a value or a variable and returns its type.",
        },
    ]

    client = FakeClient({
        "answerable": True,
        "answer": "A variable is a named memory location used to store data.",
        "source_ids": ["S1"],
    })

    result = generate_answer("what are variables?", passages, client=client)
    assert "named memory location" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 29
    assert result["citations"][0]["chunk_id"] == "c-def"
    assert result["citations"][0]["filename"] == "WebTechnology_Module 2 NOTES.pdf"


def test_simple_question_produces_direct_answer_without_introductory_padding():
    """Verify that introductory conversational padding is removed and the answer directly addresses the question."""
    passages = make_passages()
    client = FakeClient({
        "answerable": True,
        "answer": "Based on the provided PDF passages, chunking splits long documents into smaller passages [S1].",
        "source_ids": ["S1"],
    })

    result = generate_answer("Why is chunking useful?", passages, client=client)
    # Conversational meta-intro should be stripped
    assert not result["answer"].startswith("Based on the provided PDF passages")
    assert result["answer"].startswith("Chunking splits long documents into smaller passages")
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page_number"] == 2


def test_invalid_citation_is_pruned_and_not_silently_mapped_to_wrong_passage():
    """Verify that an invalid citation (e.g. S99) is pruned and NOT silently mapped to an existing passage."""
    passages = make_passages()  # Has only S1
    client = FakeClient({
        "answerable": True,
        "answer": "Chunking divides documents into smaller passages [S1]. An unverified claim [S99].",
        "source_ids": ["S1", "S99"],
    })

    result = generate_answer("What does chunking do?", passages, client=client)
    # S99 must NOT be in citations or mapped to chunk-1
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk-1"
    # S99 in-text reference should be cleaned
    assert "[S99]" not in result["answer"]
    assert "[S1]" in result["answer"]


def test_multipage_answer_preserves_citations_across_multiple_documents():
    """Verify that multi-page synthesis correctly maps citations to distinct documents and pages."""
    passages = [
        {
            "chunk_id": "c-rmf-p12",
            "filename": "nist_ai_rmf.pdf",
            "page_number": 12,
            "text": "Risk tolerance refers to an organization's readiness to bear risk.",
        },
        {
            "chunk_id": "c-csf-p30",
            "filename": "nist_csf_2.pdf",
            "page_number": 30,
            "text": "Risk tolerance informs organizational cybersecurity objectives (GV.RM-01).",
        },
    ]

    client = FakeClient({
        "answerable": True,
        "answer": (
            "In NIST AI RMF 1.0, risk tolerance is an organization's readiness to bear risk [S1]. "
            "In NIST CSF 2.0, risk tolerance informs organizational cybersecurity objectives [S2]."
        ),
        "source_ids": ["S1", "S2"],
    })

    result = generate_answer(
        "Compare how risk tolerance is defined in NIST AI RMF versus NIST CSF 2.0.",
        passages,
        client=client,
    )

    assert len(result["citations"]) == 2
    c1, c2 = result["citations"]
    assert c1["filename"] == "nist_ai_rmf.pdf"
    assert c1["page_number"] == 12
    assert c1["chunk_id"] == "c-rmf-p12"

    assert c2["filename"] == "nist_csf_2.pdf"
    assert c2["page_number"] == 30
    assert c2["chunk_id"] == "c-csf-p30"


def test_unsupported_question_abstains_clearly_with_no_citations():
    """Verify that when evidence is insufficient, the system cleanly abstains with canonical refusal."""
    passages = make_passages()
    client = FakeClient({
        "answerable": False,
        "answer": "I don't know.",
        "source_ids": [],
    })

    result = generate_answer("What is the speed of light in vacuum?", passages, client=client)
    assert result["answer"] == INSUFFICIENT_EVIDENCE
    assert result["citations"] == []


