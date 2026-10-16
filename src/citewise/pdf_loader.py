
"""Validate PDFs and extract text without running OCR."""

import hashlib
from pathlib import Path

import pymupdf

from citewise.config import (
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_BYTES,
    MIN_TEXT_CHARACTERS,
)
from citewise.models import ExtractedPage, PDFDocument, PDFLoadReport


class PDFLoadError(Exception):
    """Raised when a PDF cannot be safely processed."""


def load_pdf(path: str | Path) -> PDFLoadReport:
    """Extract page text and flag pages that may require OCR."""

    pdf_path = Path(path)

    if not pdf_path.exists() or not pdf_path.is_file():
        raise PDFLoadError(f"File does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise PDFLoadError("Input file must have a .pdf extension.")

    try:
        file_size = pdf_path.stat().st_size

        if file_size == 0:
            raise PDFLoadError("PDF file is empty.")

        if file_size > MAX_PDF_SIZE_BYTES:
            raise PDFLoadError(
                f"PDF exceeds the {MAX_PDF_SIZE_BYTES // (1024 * 1024)} MB limit."
            )

        file_hash = hashlib.sha256()

        with pdf_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                file_hash.update(block)

        pages: list[ExtractedPage] = []
        low_text_pages: list[int] = []

        with pymupdf.open(pdf_path) as pdf:
            if pdf.is_encrypted:
                raise PDFLoadError(
                    "Encrypted PDFs are not supported in this version."
                )

            page_count = len(pdf)

            if page_count == 0:
                raise PDFLoadError("PDF contains no pages.")

            if page_count > MAX_PDF_PAGES:
                raise PDFLoadError(
                    f"PDF has {page_count} pages; "
                    f"the limit is {MAX_PDF_PAGES}."
                )

            for index, page in enumerate(pdf):
                page_number = index + 1
                text = page.get_text("text").strip()
                needs_ocr = len(text) < MIN_TEXT_CHARACTERS

                if needs_ocr:
                    low_text_pages.append(page_number)

                pages.append(
                    ExtractedPage(
                        page_number=page_number,
                        text=text,
                        extraction_method="text",
                        needs_ocr=needs_ocr,
                    )
                )

        document = PDFDocument(
            document_id=file_hash.hexdigest(),
            filename=pdf_path.name,
            page_count=page_count,
            pages=pages,
            file_size_bytes=file_size,
        )

        return PDFLoadReport(
            document=document,
            empty_or_low_text_pages=low_text_pages,
        )

    except PDFLoadError:
        raise
    except (pymupdf.FileDataError, OSError, ValueError) as exc:
        raise PDFLoadError(f"Unable to read PDF: {exc}") from exc
