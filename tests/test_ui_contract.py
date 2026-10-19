"""Integration contract tests for the CiteWise Web UI and API endpoints."""

import io
import json
from http import HTTPStatus
from pathlib import Path
import pytest

from citewise.web_ui import CiteWiseRequestHandler


class FakeSocket:
    def __init__(self, request_bytes: bytes):
        self.rfile = io.BytesIO(request_bytes)
        self.wfile = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        if "b" in mode:
            if "r" in mode:
                return self.rfile
            elif "w" in mode:
                return self.wfile
        raise ValueError(f"Unsupported mode: {mode}")

    def sendall(self, data: bytes):
        self.wfile.write(data)


def simulate_request(method: str, path: str, body: bytes = b"", headers: dict[str, str] = None) -> tuple[int, dict[str, str], bytes]:
    """Helper to simulate an HTTP request through CiteWiseRequestHandler."""
    header_lines = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    if headers:
        for k, v in headers.items():
            header_lines.append(f"{k}: {v}")
    if body and "Content-Length" not in (headers or {}):
        header_lines.append(f"Content-Length: {len(body)}")
    raw_request = "\r\n".join(header_lines).encode("utf-8") + b"\r\n\r\n" + body

    sock = FakeSocket(raw_request)
    handler = CiteWiseRequestHandler(sock, ("127.0.0.1", 8080), None)

    response_bytes = sock.wfile.getvalue()
    header_end = response_bytes.find(b"\r\n\r\n")
    if header_end == -1:
        return 500, {}, response_bytes

    header_text = response_bytes[:header_end].decode("utf-8")
    resp_body = response_bytes[header_end + 4:]

    status_line = header_text.splitlines()[0]
    status_code = int(status_line.split(" ")[1])

    resp_headers = {}
    for line in header_text.splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            resp_headers[k.strip().lower()] = v.strip()

    return status_code, resp_headers, resp_body


def test_ui_index_endpoint():
    status, headers, body = simulate_request("GET", "/")
    assert status == HTTPStatus.OK
    assert "text/html" in headers.get("content-type", "")
    assert b"CiteWise" in body
    assert b"Grounded PDF Question Answering" in body


def test_ui_get_documents_endpoint(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    sample_report = reports_dir / "doc1.json"
    sample_report.write_text(
        json.dumps({
            "document": {
                "filename": "test_sample.pdf",
                "document_id": "doc123456",
                "page_count": 3,
            },
            "chunks": [{"chunk_id": "c1", "text": "sample"}],
            "chunk_count": 1,
            "status": "success",
        }),
        encoding="utf-8",
    )

    CiteWiseRequestHandler.reports_dir = reports_dir
    status, headers, body = simulate_request("GET", "/api/documents")
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert "documents" in data
    assert len(data["documents"]) == 1
    assert data["documents"][0]["filename"] == "test_sample.pdf"
    assert data["documents"][0]["page_count"] == 3
    assert data["documents"][0]["chunk_count"] == 1


def test_ui_ask_endpoint_empty_question(tmp_path: Path):
    CiteWiseRequestHandler.reports_dir = tmp_path
    payload = json.dumps({"question": "   ", "retriever": "bm25"}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert "error" in data
    assert "must not be empty" in data["error"]


def test_ui_ask_endpoint_bm25_grounded_answer(tmp_path: Path, monkeypatch):
    # Set reports_dir to actual evaluation/reports
    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    assert CiteWiseRequestHandler.reports_dir.is_dir()

    # Mock answer_generator to avoid external Gemini API call
    import citewise.web_ui as web_ui_module

    fake_result = {
        "answerable": True,
        "answer": "Chunking splits long documents into smaller segments.",
        "source_ids": ["S1"],
        "citations": [
            {
                "filename": "text_sample.pdf",
                "page_number": 2,
                "chunk_id": "b637c456e6e5-p2-c1",
            }
        ],
    }
    monkeypatch.setattr(web_ui_module, "generate_answer", lambda q, p: fake_result)

    payload = json.dumps({"question": "Why is chunking useful?", "retriever": "bm25", "top_k": 3}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "success"
    assert data["answerable"] is True
    assert "Chunking splits long documents" in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "b637c456e6e5-p2-c1"
    assert len(data["passages"]) <= 3
    assert data["passages"][0]["source_id"] == "S1"


def test_ui_ask_endpoint_honest_error_handling(monkeypatch):
    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    # Simulate API quota exhaustion or connection drop
    import citewise.web_ui as web_ui_module

    def error_generator(q, p):
        raise RuntimeError("Quota exceeded: 429 Too Many Requests")

    monkeypatch.setattr(web_ui_module, "generate_answer", error_generator)

    payload = json.dumps({"question": "Why is chunking useful?", "retriever": "bm25", "top_k": 2}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["state"] == "generation_error"
    assert data["status"] == "error"
    assert data["answerable"] is False
    assert data["answer"] == ""
    assert "429" in data["error"] or "quota" in data["error"].lower()
    # Even though LLM failed, retrieved passages are still provided to the user!
    assert len(data["passages"]) > 0


def test_ui_upload_invalid_content_type():
    status, headers, body = simulate_request("POST", "/api/upload", body=b"raw", headers={"Content-Type": "text/plain"})
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert "multipart/form-data" in data["error"]


def make_multipart_body(filename: str, content: bytes, field_name: str = "pdf") -> tuple[str, bytes]:
    boundary = "----WebKitFormBoundaryCiteWiseTest123"
    lines = [
        f"--{boundary}".encode("latin-1"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode("latin-1"),
        b"Content-Type: application/pdf",
        b"",
        content,
        f"--{boundary}--".encode("latin-1"),
        b"",
    ]
    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return content_type, body


def test_ui_upload_valid_pdf_success(tmp_path: Path, monkeypatch):
    import citewise.web_ui as web_ui_module
    from citewise.models import PDFDocument, PDFLoadReport, TextChunk
    from citewise.pipeline import IngestionResult

    uploads_dir = tmp_path / "uploads"
    reports_dir = tmp_path / "reports"
    CiteWiseRequestHandler.uploads_dir = uploads_dir
    CiteWiseRequestHandler.reports_dir = reports_dir

    fake_report = PDFLoadReport(
        document=PDFDocument(
            document_id="mock12345678",
            filename="mock_doc.pdf",
            page_count=2,
            pages=[],
            file_size_bytes=100,
        ),
        empty_or_low_text_pages=[],
    )
    fake_chunks = [
        TextChunk(
            chunk_id="mock12345678-p1-c1",
            document_id="mock12345678",
            page_number=1,
            chunk_index=1,
            text="Mock extracted text",
            extraction_method="direct",
        )
    ]
    monkeypatch.setattr(
        web_ui_module,
        "process_pdf",
        lambda p: IngestionResult(report=fake_report, chunks=fake_chunks),
    )

    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    content_type, body = make_multipart_body("mock_doc.pdf", pdf_bytes)

    status, headers, resp_body = simulate_request(
        "POST", "/api/upload", body=body, headers={"Content-Type": content_type}
    )
    assert status == HTTPStatus.OK
    data = json.loads(resp_body.decode("utf-8"))
    assert data["status"] == "success"
    assert data["filename"] == "mock_doc.pdf"
    assert data["pages"] == 2
    assert data["chunks"] == 1
    assert (uploads_dir / "mock_doc.pdf").is_file()
    assert (reports_dir / "mock12345678.json").is_file()


def test_ui_upload_non_pdf_rejected(tmp_path: Path):
    CiteWiseRequestHandler.uploads_dir = tmp_path / "uploads"
    CiteWiseRequestHandler.reports_dir = tmp_path / "reports"

    content_type, body = make_multipart_body("script.py", b"print('hello')")
    status, headers, resp_body = simulate_request(
        "POST", "/api/upload", body=body, headers={"Content-Type": content_type}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(resp_body.decode("utf-8"))
    assert data["status"] == "error"
    assert "Only .pdf files are accepted" in data["error"]


def test_ui_upload_missing_file_payload(tmp_path: Path):
    CiteWiseRequestHandler.uploads_dir = tmp_path / "uploads"
    CiteWiseRequestHandler.reports_dir = tmp_path / "reports"

    # Multipart without pdf part
    boundary = "----WebKitFormBoundaryCiteWiseTest123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="other"\r\n\r\n'
        f"some data\r\n"
        f"--{boundary}--\r\n"
    ).encode("latin-1")
    content_type = f"multipart/form-data; boundary={boundary}"

    status, headers, resp_body = simulate_request(
        "POST", "/api/upload", body=body, headers={"Content-Type": content_type}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(resp_body.decode("utf-8"))
    assert "Missing valid PDF file in upload" in data["error"]


def test_ui_upload_ingestion_failure_reporting(tmp_path: Path, monkeypatch):
    import citewise.web_ui as web_ui_module
    from citewise.pdf_loader import PDFLoadError

    CiteWiseRequestHandler.uploads_dir = tmp_path / "uploads"
    CiteWiseRequestHandler.reports_dir = tmp_path / "reports"

    def fail_ingestion(p):
        raise PDFLoadError("PDF file is corrupted or password encrypted.")

    monkeypatch.setattr(web_ui_module, "process_pdf", fail_ingestion)

    content_type, body = make_multipart_body("corrupt.pdf", b"corrupted data")
    status, headers, resp_body = simulate_request(
        "POST", "/api/upload", body=body, headers={"Content-Type": content_type}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(resp_body.decode("utf-8"))
    assert data["status"] == "error"
    assert "PDF file is corrupted" in data["error"]


def test_ui_ask_endpoint_invalid_json():
    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=b"{malformed json", headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert "Invalid JSON payload" in data["error"]


def test_ui_ask_endpoint_zero_reports_found(tmp_path: Path):
    empty_reports = tmp_path / "empty_reports"
    empty_reports.mkdir()
    CiteWiseRequestHandler.reports_dir = empty_reports

    payload = json.dumps({"question": "Any question?", "retriever": "bm25"}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert "No indexed PDF reports found" in data["error"]


def test_ui_ask_endpoint_missing_semantic_index_for_hybrid(tmp_path: Path):
    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = tmp_path / "nonexistent_index.json"

    payload = json.dumps({"question": "Any question?", "retriever": "hybrid"}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert "Semantic index not found" in data["error"]


def test_ui_ask_endpoint_abstention_handling(monkeypatch):
    import citewise.web_ui as web_ui_module
    from citewise.answer_generator import INSUFFICIENT_EVIDENCE

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    fake_abstention = {
        "answerable": False,
        "answer": INSUFFICIENT_EVIDENCE,
        "source_ids": [],
        "citations": [],
    }
    monkeypatch.setattr(web_ui_module, "generate_answer", lambda q, p: fake_abstention)

    payload = json.dumps({"question": "What is NIST risk tolerance?", "retriever": "bm25"}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["state"] == "insufficient_evidence"
    assert data["answerable"] is False
    assert data["answer"] == INSUFFICIENT_EVIDENCE
    assert data["citations"] == []
    assert len(data["passages"]) > 0


def test_ui_ask_endpoint_missing_api_key_preserves_passages(monkeypatch):
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    def missing_key_generator(q, p):
        raise ValueError("GEMINI_API_KEY environment variable is not set. Live generation is disabled.")

    monkeypatch.setattr(web_ui_module, "generate_answer", missing_key_generator)

    payload = json.dumps({"question": "What is the NIST AI Risk Management Framework?", "retriever": "bm25"}).encode("utf-8")
    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["state"] == "generation_error"
    assert data["status"] == "error"
    assert data["answerable"] is False
    assert data["answer"] == ""
    assert "GEMINI_API_KEY" in data["error"]
    # Evidence passages are strictly preserved for user display
    assert len(data["passages"]) > 0
    assert data["passages"][0]["source_id"] == "S1"
    assert "filename" in data["passages"][0]
    assert "page_number" in data["passages"][0]


def test_ui_user_workspace_isolation(tmp_path: Path):
    """Prove that benchmark reports in evaluation/reports do not enter ordinary user workspace."""
    user_reports = tmp_path / "user_reports"
    user_reports.mkdir()

    # User uploads a single doc
    user_doc_report = user_reports / "user_doc.json"
    user_doc_report.write_text(
        json.dumps({
            "document": {
                "filename": "my_uploaded_notes.pdf",
                "document_id": "user12345",
                "page_count": 5,
            },
            "chunks": [
                {
                    "chunk_id": "user12345-p1-c1",
                    "document_id": "user12345",
                    "page_number": 1,
                    "text": "These are private user uploaded notes about web engineering.",
                }
            ],
            "chunk_count": 1,
            "status": "success",
        }),
        encoding="utf-8",
    )

    CiteWiseRequestHandler.reports_dir = user_reports
    status, headers, body = simulate_request("GET", "/api/documents")
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    filenames = [d["filename"] for d in data["documents"]]

    assert "my_uploaded_notes.pdf" in filenames
    # Verify zero benchmark PDFs entered the user workspace
    assert "nist_ai_rmf.pdf" not in filenames
    assert "nist_csf_2.pdf" not in filenames
    assert "scan_sample.pdf" not in filenames
    assert "text_sample.pdf" not in filenames


def test_ui_document_selection_filtering(tmp_path: Path, monkeypatch):
    """Test that selected_documents restricts retrieval strictly to the chosen files."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    (reports_dir / "doc_a.json").write_text(
        json.dumps({
            "document": {"filename": "DocA.pdf", "document_id": "idA", "page_count": 1},
            "chunks": [
                {"chunk_id": "idA-p1-c1", "document_id": "idA", "page_number": 1, "text": "Alpha chunk text content"}
            ],
            "chunk_count": 1,
            "status": "success",
        }),
        encoding="utf-8",
    )
    (reports_dir / "doc_b.json").write_text(
        json.dumps({
            "document": {"filename": "DocB.pdf", "document_id": "idB", "page_count": 1},
            "chunks": [
                {"chunk_id": "idB-p1-c1", "document_id": "idB", "page_number": 1, "text": "Beta chunk text content"}
            ],
            "chunk_count": 1,
            "status": "success",
        }),
        encoding="utf-8",
    )

    CiteWiseRequestHandler.reports_dir = reports_dir

    import citewise.web_ui as web_ui_module
    captured_passages = []

    def mock_generator(q, passages):
        captured_passages.extend(passages)
        return {
            "answerable": True,
            "answer": "Answer based only on DocA.",
            "source_ids": ["S1"],
            "citations": [{"filename": "DocA.pdf", "page_number": 1, "chunk_id": "idA-p1-c1"}],
        }

    monkeypatch.setattr(web_ui_module, "generate_answer", mock_generator)

    payload = json.dumps({
        "question": "Alpha text content?",
        "retriever": "bm25",
        "selected_documents": ["DocA.pdf"],
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["state"] == "answered"
    assert len(data["passages"]) == 1
    assert data["passages"][0]["filename"] == "DocA.pdf"
    assert all(p["filename"] == "DocA.pdf" for p in captured_passages)


def test_ui_evidence_gate_rejection_zero_generation_calls(monkeypatch):
    """Verify that a question rejected by the evidence gate triggers ZERO calls to generate_answer."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    generator_call_count = 0

    def mock_generator(q, p):
        nonlocal generator_call_count
        generator_call_count += 1
        return {"answerable": True, "answer": "Should not be called", "citations": []}

    monkeypatch.setattr(web_ui_module, "generate_answer", mock_generator)

    # An out-of-context question that has 0 content word overlap with the NIST/sample documents
    out_of_context_question = "What is the average rainfall in the Amazon rainforest during monsoon season?"
    payload = json.dumps({
        "question": out_of_context_question,
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    # Evidence gate rejected query locally
    assert data["state"] == "insufficient_evidence"
    assert data["answerable"] is False
    assert data["gate"]["decision"] == "reject"
    assert "Zero lexical overlap" in data["gate"]["reason"]

    # CRITICAL: Verify generator was never invoked
    assert generator_call_count == 0, f"Expected 0 generation calls, got {generator_call_count}"


def test_ui_simulated_503_error_distinguishable_from_abstention(monkeypatch):
    """Verify that a simulated 503 error returns generation_error, NOT insufficient_evidence."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    def simulate_503(q, p):
        raise ConnectionError("503 UNAVAILABLE: The model is currently overloaded. Please try again later.")

    monkeypatch.setattr(web_ui_module, "generate_answer", simulate_503)

    payload = json.dumps({
        "question": "What is the primary function of the NIST AI RMF Core?",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    # Error semantics must be generation_error, NOT insufficient_evidence
    assert data["state"] == "generation_error"
    assert data["status"] == "error"
    assert "503" in data["error"]
    assert "temporarily unavailable" in data["error"].lower()
    # Must NOT return insufficient evidence string as answer
    assert data["answer"] == ""
    assert data["answerable"] is False
    # Must preserve passages
    assert len(data["passages"]) > 0
    assert data["passages"][0]["source_id"] == "S1"


@pytest.mark.parametrize("invalid_top_k", [0, -1, -5, "abc", True, False, []])
def test_ui_ask_endpoint_invalid_top_k(invalid_top_k):
    """Verify API boundary rejects invalid top_k with 400 Bad Request instead of retrieval traceback."""
    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")

    payload = json.dumps({
        "question": "What is the primary function of the NIST AI RMF Core?",
        "retriever": "bm25",
        "top_k": invalid_top_k,
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.BAD_REQUEST
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "error"
    assert "Invalid top_k: top_k must be a positive integer" in data["error"]


@pytest.mark.parametrize("top_k_val", [3, 5, 8, "3", "5", "8"])
@pytest.mark.parametrize("retriever", ["bm25", "hybrid"])
def test_ui_ask_endpoint_supported_top_k_options(monkeypatch, top_k_val, retriever):
    """Verify all UI dropdown top_k values (3, 5, 8) succeed for both BM25 and Hybrid."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    # Mock generator so no external Gemini calls are made
    monkeypatch.setattr(
        web_ui_module,
        "generate_answer",
        lambda q, p: {"answerable": True, "answer": "Mocked test answer", "source_ids": ["S1"]},
    )

    payload = json.dumps({
        "question": "What is the primary function of the NIST AI RMF Core?",
        "retriever": retriever,
        "top_k": top_k_val,
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "success" or data.get("state") in ("success", "insufficient_evidence")
    expected_k = int(top_k_val)
    assert len(data["passages"]) <= expected_k


def test_ui_ask_hybrid_top_3_with_unembedded_document(tmp_path: Path, monkeypatch):
    """Verify Hybrid search with Top 3 does not crash when selected document has no semantic embeddings."""
    import citewise.web_ui as web_ui_module

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    doc_report = reports_dir / "user_doc.json"
    doc_report.write_text(
        json.dumps({
            "document": {
                "filename": "WebTechnology_Module 2 NOTES.pdf",
                "document_id": "userdoc123",
                "page_count": 29,
            },
            "chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "userdoc123",
                    "page_number": 29,
                    "text": "Variables: A variable is named memory location to store data.",
                    "filename": "WebTechnology_Module 2 NOTES.pdf",
                },
                {
                    "chunk_id": "c2",
                    "document_id": "userdoc123",
                    "page_number": 26,
                    "text": "JavaScript uses variables and functions.",
                    "filename": "WebTechnology_Module 2 NOTES.pdf",
                },
            ],
            "chunk_count": 2,
            "status": "success",
        }),
        encoding="utf-8",
    )

    # Empty semantic index (like when a user uploads a new PDF that has not been embedded)
    index_file = tmp_path / "semantic_index.json"
    index_file.write_text(
        json.dumps({
            "index_version": 1,
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "items": [],
        }),
        encoding="utf-8",
    )

    CiteWiseRequestHandler.reports_dir = reports_dir
    CiteWiseRequestHandler.index_path = index_file

    monkeypatch.setattr(
        web_ui_module,
        "generate_answer",
        lambda q, p: {"answerable": True, "answer": "A variable is named memory location.", "source_ids": ["S1"]},
    )

    payload = json.dumps({
        "question": "what are variables?",
        "retriever": "hybrid",
        "top_k": 3,
        "selected_documents": ["WebTechnology_Module 2 NOTES.pdf"],
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "success"
    # Ensure chunk from page 29 is returned
    assert len(data["passages"]) >= 1
    assert data["passages"][0]["page_number"] == 29
    assert data["passages"][0]["chunk_id"] == "c1"


def test_ui_ask_endpoint_gate_pass_with_model_abstention_semantics(monkeypatch):
    """Verify that when the gate passes but the model abstains, state is insufficient_evidence
    and answer contains the model's abstention text rather than gate metrics."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    # Mock generator returning grounded abstention
    monkeypatch.setattr(
        web_ui_module,
        "generate_answer",
        lambda q, p: {"answerable": False, "answer": web_ui_module.INSUFFICIENT_EVIDENCE, "citations": []},
    )

    payload = json.dumps({
        "question": "What are the six Functions that organize outcomes in NIST CSF 2.0 Core?",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert data["state"] == "insufficient_evidence"
    assert data["status"] == "insufficient_evidence"
    assert data["answerable"] is False
    assert web_ui_module.INSUFFICIENT_EVIDENCE in data["answer"]
    # Gate passed, but model abstained
    assert data["gate"]["decision"] in ("pass", "borderline")
    assert len(data["passages"]) > 0


def test_ui_ask_endpoint_mocked_429_quota_error(monkeypatch):
    """Verify 429 quota exhaustion returns state: generation_error with preserved passages."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    def raise_429(q, p):
        raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for project")

    monkeypatch.setattr(web_ui_module, "generate_answer", raise_429)

    payload = json.dumps({
        "question": "What is the primary function of the NIST AI RMF Core?",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert data["state"] == "generation_error"
    assert data["status"] == "error"
    assert "429" in data["error"]
    assert "rate limit or quota exceeded" in data["error"].lower()
    assert data["answer"] == ""
    assert data["answerable"] is False
    assert len(data["passages"]) > 0


def test_ui_ask_endpoint_mocked_timeout_and_credential_sanitization(monkeypatch):
    """Verify timeout and arbitrary errors sanitize credentials and return generation_error."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    def raise_err(q, p):
        raise Exception("API failure for endpoint https://api.google.com/v1?key=AIzaSyDUMMYKEY1234567890abcdef1234567890: request timed out")

    monkeypatch.setattr(web_ui_module, "generate_answer", raise_err)

    payload = json.dumps({
        "question": "What is the primary function of the NIST AI RMF Core?",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert data["state"] == "generation_error"
    assert data["status"] == "error"
    assert "AIzaSyDUMMYKEY" not in data["error"]
    assert "timed out" in data["error"].lower()

    # Now verify arbitrary non-standard error redacts API key
    def raise_with_key(q, p):
        raise Exception("Failed with key AIzaSyDUMMYKEY1234567890abcdef1234567890")

    monkeypatch.setattr(web_ui_module, "generate_answer", raise_with_key)
    status2, headers2, body2 = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    data2 = json.loads(body2.decode("utf-8"))
    assert "[REDACTED_API_KEY]" in data2["error"]
    assert "AIzaSyDUMMYKEY" not in data2["error"]


def test_ui_ask_endpoint_rejected_query_skips_generation(monkeypatch):
    """Verify that when evidence gate rejects a query, generate_answer is called 0 times."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    call_count = {"count": 0}

    def fake_generate(q, p):
        call_count["count"] += 1
        return {"answer": "Should not be called", "citations": []}

    monkeypatch.setattr(web_ui_module, "generate_answer", fake_generate)

    # Completely unrelated query with zero lexical overlap
    payload = json.dumps({
        "question": "What is the average rainfall in the Amazon rainforest during monsoon season?",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert call_count["count"] == 0
    assert data["state"] == "insufficient_evidence"
    assert data["gate"]["decision"] == "reject"


def test_ui_ask_endpoint_multi_page_answer_and_citations(monkeypatch):
    """Verify multi-page questions receive citations spanning multiple pages."""
    import citewise.web_ui as web_ui_module

    CiteWiseRequestHandler.reports_dir = Path("evaluation/reports")
    CiteWiseRequestHandler.index_path = Path("data/semantic_index.json")

    # Mock generator returning citations from two different pages
    monkeypatch.setattr(
        web_ui_module,
        "generate_answer",
        lambda q, p: {
            "answerable": True,
            "answer": "This fact is based on page 8 and page 20.",
            "citations": [
                {"filename": "nist_csf_2.pdf", "page_number": 8, "chunk_id": "c_p8"},
                {"filename": "nist_csf_2.pdf", "page_number": 20, "chunk_id": "c_p20"},
            ],
            "source_ids": ["S1", "S2"],
        },
    )

    payload = json.dumps({
        "question": "Compare the core hierarchy and governance tiers in NIST CSF 2.0",
        "retriever": "bm25",
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert data["state"] == "answered"
    assert data["status"] == "success"
    assert len(data["citations"]) == 2
    pages = {c["page_number"] for c in data["citations"]}
    assert pages == {8, 20}


def test_ui_ask_endpoint_javascript_invention_positive_test(monkeypatch):
    """Verify Who invented JavaScript is recognized as answerable from WebTechnology_Module 2 NOTES.pdf."""
    import citewise.web_ui as web_ui_module

    # Test report containing the Brendan Eich chunk
    report_path = Path("user_data/reports/bc3b4bc509c8.json")
    if not report_path.is_file():
        pytest.skip("User data report not found.")

    CiteWiseRequestHandler.reports_dir = Path("user_data/reports")

    monkeypatch.setattr(
        web_ui_module,
        "generate_answer",
        lambda q, p: {
            "answerable": True,
            "answer": "JavaScript was invented by Brendan Eich.",
            "citations": [{"filename": "WebTechnology_Module 2 NOTES.pdf", "page_number": 24, "chunk_id": "bc3b4bc509c8-p24-c2"}],
            "source_ids": ["S1"],
        },
    )

    payload = json.dumps({
        "question": "Who invented JavaScript?",
        "retriever": "bm25",
        "selected_documents": ["WebTechnology_Module 2 NOTES.pdf"],
    }).encode("utf-8")

    status, headers, body = simulate_request(
        "POST", "/api/ask", body=payload, headers={"Content-Type": "application/json"}
    )
    assert status == HTTPStatus.OK
    data = json.loads(body.decode("utf-8"))

    assert data["gate"]["decision"] == "pass"
    assert data["state"] == "answered"
    assert "Brendan Eich" in data["answer"]
    assert any("Brendan Eich" in p["text"] for p in data["passages"])


def test_ui_html_math_and_code_formatting_contract():
    """Verify that HTML template includes KaTeX resources, math styling, and code block headers."""
    status, headers, body = simulate_request("GET", "/")
    assert status == HTTPStatus.OK
    html_text = body.decode("utf-8")

    # Math formatting checks
    assert "katex.min.css" in html_text
    assert "katex.min.js" in html_text
    assert ".math-display" in html_text
    assert ".math-inline" in html_text
    assert "formatSimpleMath" in html_text
    assert "renderMathExpression" in html_text

    # Code formatting checks
    assert ".code-block-wrapper" in html_text
    assert ".code-block-header" in html_text
    assert "___TOKEN_CODE_BLOCK_" in html_text


def test_ui_unsupported_passages_transparency_contract():
    """Verify that HTML template renders unsupported passages in a collapsed details section."""
    status, headers, body = simulate_request("GET", "/")
    assert status == HTTPStatus.OK
    html_text = body.decode("utf-8")

    assert "unsupported-passages-details" in html_text
    assert "Retrieved passages (not sufficient to answer)" in html_text
    assert "preserved here for transparency" in html_text


def test_ui_citation_links_validation_contract():
    """Verify that HTML template includes unverified-citation styling and validates citation IDs against retrieved passages."""
    status, headers, body = simulate_request("GET", "/")
    assert status == HTTPStatus.OK
    html_text = body.decode("utf-8")

    assert ".unverified-citation" in html_text
    assert "validSourceIds" in html_text
    assert "unverified" in html_text





