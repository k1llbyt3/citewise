
"""Data structures returned by the PDF loader."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: str
    needs_ocr: bool


@dataclass
class PDFDocument:
    document_id: str
    filename: str
    page_count: int
    pages: list[ExtractedPage]
    file_size_bytes: int


@dataclass
class PDFLoadReport:
    document: PDFDocument
    empty_or_low_text_pages: list[int]

@dataclass
class TextChunk:
    """A chunk of extracted text with source metadata."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    extraction_method: str