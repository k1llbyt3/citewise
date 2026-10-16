
"""OCR fallback for pages with little or no extractable text."""

from pathlib import Path

import pymupdf

from citewise.models import PDFLoadReport


class OCRProcessingError(Exception):
    """Raised when OCR cannot process a PDF page."""


def extract_ocr_text(
    pdf_path: str | Path,
    page_number: int,
    language: str = "eng",
    dpi: int = 150,
) -> str:
    """Extract text from one PDF page using Tesseract OCR.

    Args:
        pdf_path: Path to the original PDF.
        page_number: One-based page number.
        language: Installed Tesseract language code.
        dpi: Rendering resolution used for OCR.

    Returns:
        Recognized text, stripped of surrounding whitespace.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise OCRProcessingError(f"PDF does not exist: {path}")

    if page_number < 1:
        raise OCRProcessingError("Page number must be at least 1.")

    if dpi < 72 or dpi > 300:
        raise OCRProcessingError("DPI must be between 72 and 300.")

    try:
        with pymupdf.open(path) as pdf:
            if pdf.is_encrypted:
                raise OCRProcessingError(
                    "Encrypted PDFs are not supported."
                )

            if page_number > len(pdf):
                raise OCRProcessingError(
                    f"Page {page_number} does not exist in this PDF."
                )

            page = pdf[page_number - 1]

            # PyMuPDF uses Tesseract to recognize text on the rendered page.
           
            text_page = page.get_textpage_ocr(
    language=language,
    dpi=dpi,
    full=True,
    tessdata=r"C:\Program Files\Tesseract-OCR\tessdata",
)


            return page.get_text(
                "text",
                textpage=text_page,
            ).strip()

    except OCRProcessingError:
        raise
    except (pymupdf.FileDataError, OSError, RuntimeError, ValueError) as exc:
        raise OCRProcessingError(
            f"OCR failed on page {page_number}: {exc}. "
            "Check that Tesseract and the requested language data "
            "are installed and accessible."
        ) from exc


def apply_ocr_fallback(
    pdf_path: str | Path,
    report: PDFLoadReport,
    language: str = "eng",
    dpi: int = 150,
) -> PDFLoadReport:
    """OCR only pages flagged by the PDF loader.

    Successful OCR replaces the page's text and updates its extraction
    method. Pages that remain unreadable stay flagged for investigation.
    """
    for page in report.document.pages:
        if not page.needs_ocr:
            continue

        text = extract_ocr_text(
            pdf_path=pdf_path,
            page_number=page.page_number,
            language=language,
            dpi=dpi,
        )

        if text:
            page.text = text
            page.extraction_method = "ocr"
            page.needs_ocr = False

    report.empty_or_low_text_pages = [
        page.page_number
        for page in report.document.pages
        if page.needs_ocr
    ]

    return report
