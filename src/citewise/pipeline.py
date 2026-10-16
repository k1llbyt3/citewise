"""Orchestrate PDF extraction, OCR fallback, and text chunking."""

from dataclasses import dataclass
from pathlib import Path

from citewise.chunker import chunk_document
from citewise.models import PDFLoadReport, TextChunk
from citewise.ocr import apply_ocr_fallback,OCRProcessingError
from citewise.pdf_loader import load_pdf,PDFLoadError


@dataclass
class IngestionResult:
    """Results produced by the CiteWise ingestion pipeline."""

    report: PDFLoadReport
    chunks: list[TextChunk]


def process_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1000,
    overlap: int = 150,
    ocr_language: str = "eng",
    ocr_dpi: int = 150,
) -> IngestionResult:
    """Process a PDF from initial extraction through chunk generation.

    Steps:
    1. Validate the PDF and extract available text.
    2. Run OCR on pages flagged as having insufficient text.
    3. Split extracted text into chunks while preserving page metadata.

    Raises:
        PDFLoadError: If the input PDF cannot be loaded.
        OCRProcessingError: If OCR fails on a page that needs it.
        ValueError: If chunk settings are invalid.
    """
    report = load_pdf(pdf_path)

    report = apply_ocr_fallback(
        pdf_path=pdf_path,
        report=report,
        language=ocr_language,
        dpi=ocr_dpi,
    )

    chunks = chunk_document(
        document=report.document,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return IngestionResult(
        report=report,
        chunks=chunks,
    )
