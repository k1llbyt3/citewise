
import pymupdf
import pytest

from citewise.ocr import (
    OCRProcessingError,
    apply_ocr_fallback,
    extract_ocr_text,
)
from citewise.pdf_loader import load_pdf


def create_pdf(path, pages):
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        pdf.save(path)
    return path


def test_rejects_missing_pdf(tmp_path):
    with pytest.raises(OCRProcessingError, match="does not exist"):
        extract_ocr_text(tmp_path / "missing.pdf", 1)


def test_rejects_invalid_page_number(tmp_path):
    path = create_pdf(tmp_path / "sample.pdf", ["Some text."])

    with pytest.raises(OCRProcessingError, match="at least 1"):
        extract_ocr_text(path, 0)


def test_rejects_invalid_dpi(tmp_path):
    path = create_pdf(tmp_path / "sample.pdf", ["Some text."])

    with pytest.raises(OCRProcessingError, match="DPI"):
        extract_ocr_text(path, 1, dpi=500)


def test_rejects_page_outside_document(tmp_path):
    path = create_pdf(tmp_path / "sample.pdf", ["Some text."])

    with pytest.raises(OCRProcessingError, match="does not exist"):
        extract_ocr_text(path, 2)


def test_ocr_updates_only_flagged_pages(tmp_path, monkeypatch):
    path = create_pdf(
        tmp_path / "mixed.pdf",
        [
            "",
            "This page already has enough extractable text.",
        ],
    )

    report = load_pdf(path)
    calls = []

    def fake_ocr(pdf_path, page_number, language="eng", dpi=150):
        calls.append(page_number)
        return "Recognized text from scanned page."

    monkeypatch.setattr("citewise.ocr.extract_ocr_text", fake_ocr)

    result = apply_ocr_fallback(path, report)

    assert calls == [1]
    assert result.document.pages[0].text == (
        "Recognized text from scanned page."
    )
    assert result.document.pages[0].extraction_method == "ocr"
    assert result.document.pages[0].needs_ocr is False

    assert result.document.pages[1].extraction_method == "text"
    assert result.empty_or_low_text_pages == []


def test_empty_ocr_result_remains_flagged(tmp_path, monkeypatch):
    path = create_pdf(tmp_path / "blank.pdf", [""])
    report = load_pdf(path)

    monkeypatch.setattr(
        "citewise.ocr.extract_ocr_text",
        lambda *args, **kwargs: "",
    )

    result = apply_ocr_fallback(path, report)

    assert result.document.pages[0].needs_ocr is True
    assert result.document.pages[0].extraction_method == "text"
    assert result.empty_or_low_text_pages == [1]
