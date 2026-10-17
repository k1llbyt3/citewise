
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
