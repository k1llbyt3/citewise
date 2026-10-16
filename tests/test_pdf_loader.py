
from pathlib import Path

import pymupdf
import pytest

from citewise.pdf_loader import PDFLoadError, load_pdf


def create_pdf(path: Path, pages: list[str]) -> Path:
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        pdf.save(path)
    return path


def test_extracts_text_and_page_numbers(tmp_path):
    path = create_pdf(
        tmp_path / "sample.pdf",
        [
            "CiteWise extracts text from PDFs.",
            "RAG retrieves relevant document passages.",
        ],
    )

    report = load_pdf(path)

    assert report.document.page_count == 2
    assert report.document.pages[0].page_number == 1
    assert "CiteWise" in report.document.pages[0].text
    assert report.document.pages[1].page_number == 2
    assert report.document.pages[0].extraction_method == "text"
    assert report.empty_or_low_text_pages == []


def test_flags_blank_page_for_ocr(tmp_path):
    path = create_pdf(
        tmp_path / "blank.pdf",
        ["", "This page contains enough text to be extracted."],
    )

    report = load_pdf(path)

    assert report.document.pages[0].needs_ocr is True
    assert report.document.pages[1].needs_ocr is False
    assert report.empty_or_low_text_pages == [1]


def test_same_content_has_same_document_id(tmp_path):
    first = create_pdf(tmp_path / "first.pdf", ["Same document content."])
    second = tmp_path / "second.pdf"

    # Copy the exact bytes, including PDF metadata.
    second.write_bytes(first.read_bytes())

    assert load_pdf(first).document.document_id == (
        load_pdf(second).document.document_id
    )


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(PDFLoadError, match="does not exist"):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_extension_is_rejected(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(PDFLoadError, match="\\.pdf"):
        load_pdf(file_path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")

    with pytest.raises(PDFLoadError, match="empty"):
        load_pdf(path)


def test_invalid_pdf_is_rejected(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"This is not a valid PDF document.")

    with pytest.raises(PDFLoadError):
        load_pdf(path)


def test_rejects_pdf_over_size_limit(tmp_path, monkeypatch):
    import citewise.pdf_loader as loader

    path = create_pdf(tmp_path / "large.pdf", ["A valid PDF for testing."])

    # Lower the limit for this test instead of creating a 25 MB file.
    monkeypatch.setattr(loader, "MAX_PDF_SIZE_BYTES", 100)

    with pytest.raises(PDFLoadError, match="limit"):
        load_pdf(path)


def test_rejects_pdf_over_page_limit(tmp_path, monkeypatch):
    import citewise.pdf_loader as loader

    path = create_pdf(
        tmp_path / "too_many_pages.pdf",
        ["Page one.", "Page two.", "Page three."],
    )

    # Make three pages exceed the test limit.
    monkeypatch.setattr(loader, "MAX_PDF_PAGES", 2)

    with pytest.raises(PDFLoadError, match="pages"):
        load_pdf(path)


def test_rejects_encrypted_pdf(tmp_path):
    path = tmp_path / "encrypted.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "This PDF will be encrypted.")

        pdf.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )

    with pytest.raises(PDFLoadError, match="Encrypted"):
        load_pdf(path)


def test_text_threshold_boundary(tmp_path):
    path = create_pdf(
        tmp_path / "threshold.pdf",
        [
            "1234567890123456789",   # 19 characters
            "12345678901234567890",  # 20 characters
        ],
    )

    report = load_pdf(path)

    assert report.document.pages[0].needs_ocr is True
    assert report.document.pages[1].needs_ocr is False
    assert report.empty_or_low_text_pages == [1]
