
import pymupdf
import pytest

import citewise.pipeline as pipeline
from citewise.chunker import chunk_document
from citewise.models import ExtractedPage, PDFDocument
from citewise.pipeline import process_pdf


def create_text_pdf(path):
    """Create a small text-based PDF for testing."""
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "CiteWise pipeline test document with enough readable text.",
            fontsize=12,
        )
        pdf.save(path)


def test_process_text_pdf(tmp_path):
    pdf_path = tmp_path / "text_sample.pdf"
    create_text_pdf(pdf_path)

    result = process_pdf(pdf_path)

    assert result.report.document.filename == "text_sample.pdf"
    assert result.report.document.page_count == 1
    assert len(result.chunks) >= 1
    assert any(
        "CiteWise pipeline test document" in chunk.text
        for chunk in result.chunks
    )
    assert all(
        chunk.document_id == result.report.document.document_id
        for chunk in result.chunks
    )


def test_process_pdf_passes_chunk_settings(tmp_path):
    pdf_path = tmp_path / "text_sample.pdf"
    create_text_pdf(pdf_path)

    result = process_pdf(pdf_path, chunk_size=30, overlap=5)

    assert len(result.chunks) > 1
    assert all(len(chunk.text) <= 30 for chunk in result.chunks)


def test_process_pdf_runs_ocr_fallback(tmp_path, monkeypatch):
    pdf_path = tmp_path / "text_sample.pdf"
    create_text_pdf(pdf_path)

    called = {}

    def fake_ocr(pdf_path, report, language, dpi):
        called["language"] = language
        called["dpi"] = dpi
        return report

    monkeypatch.setattr(pipeline, "apply_ocr_fallback", fake_ocr)

    process_pdf(pdf_path, ocr_language="eng", ocr_dpi=150)

    assert called == {"language": "eng", "dpi": 150}


def test_invalid_chunk_settings_propagate(tmp_path):
    pdf_path = tmp_path / "text_sample.pdf"
    create_text_pdf(pdf_path)

    with pytest.raises(ValueError):
        process_pdf(pdf_path, chunk_size=0)


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(pipeline.PDFLoadError):
        process_pdf(tmp_path / "missing.pdf")
