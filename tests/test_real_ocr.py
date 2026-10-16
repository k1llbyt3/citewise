
"""Integration test using the real Tesseract OCR engine."""

import pymupdf

from citewise.ocr import extract_ocr_text
from citewise.pdf_loader import load_pdf


def test_real_tesseract_ocr_reads_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "ocr_test_generated.pdf"

    # Create a page with text, then render it as an image.
    with pymupdf.open() as source:
        page = source.new_page(width=595, height=842)
        page.insert_text(
            (72, 120),
            "CiteWise OCR Test. This document contains scanned text.",
            fontsize=20,
        )
        page.insert_text(
            (72, 170),
            "The quick brown fox jumps over the lazy dog.",
            fontsize=16,
        )

        image = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False,
        ).tobytes("png")

    # Create a PDF containing only the image, without selectable text.
    with pymupdf.open() as scanned:
        page = scanned.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=image)
        scanned.save(pdf_path)

    # Ordinary extraction should flag the image-only page.
    report = load_pdf(pdf_path)

    assert report.document.page_count == 1
    assert report.empty_or_low_text_pages == [1]
    assert report.document.pages[0].needs_ocr is True

    # Run actual Tesseract OCR.
    text = extract_ocr_text(pdf_path, page_number=1)

    assert "CiteWise OCR Test" in text
    assert "scanned text" in text.lower()
