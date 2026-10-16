
import pytest

from citewise.chunker import chunk_document, normalize_text
from citewise.models import ExtractedPage, PDFDocument


def make_document(page_texts: list[str]) -> PDFDocument:
    return PDFDocument(
        document_id="abcdef1234567890",
        filename="sample.pdf",
        page_count=len(page_texts),
        pages=[
            ExtractedPage(
                page_number=index,
                text=text,
                extraction_method="text",
                needs_ocr=False,
            )
            for index, text in enumerate(page_texts, start=1)
        ],
        file_size_bytes=1024,
    )


def test_normalize_text():
    assert normalize_text("  Hello   world \r\n\r\nTest  ") == (
        "Hello world\n\nTest"
    )


def test_short_page_produces_one_chunk():
    document = make_document(["This is a short page."])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == "This is a short page."
    assert chunks[0].page_number == 1
    assert chunks[0].document_id == document.document_id


def test_long_page_produces_multiple_chunks():
    document = make_document(["word " * 100])

    chunks = chunk_document(document, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_empty_pages_are_skipped():
    document = make_document(["", "   ", "Actual page content"])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3


def test_chunks_preserve_extraction_method():
    document = make_document(["Recognized by OCR"])
    document.pages[0].extraction_method = "ocr"

    chunks = chunk_document(document)

    assert chunks[0].extraction_method == "ocr"


def test_invalid_chunk_settings():
    document = make_document(["Some text"])

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=100, overlap=100)
