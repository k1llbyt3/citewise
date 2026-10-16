
"""Normalize extracted PDF text and split it into metadata-rich chunks."""

import re

from citewise.models import PDFDocument, TextChunk



def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing spaces from each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Normalize repeated spaces and tabs within lines.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse three or more consecutive newlines to two.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def _find_chunk_end(text: str, start: int, target: int) -> int:
    """Prefer a natural boundary near the target chunk length."""
    if target >= len(text):
        return len(text)

    # Prefer paragraph, line, or sentence boundaries when nearby.
    minimum = start + int((target - start) * 0.7)
    candidates = [
        text.rfind("\n\n", minimum, target),
        text.rfind("\n", minimum, target),
        text.rfind(". ", minimum, target),
        text.rfind(" ", minimum, target),
    ]

    boundary = max(candidates)
    if boundary >= minimum:
        # Keep sentence punctuation in the preceding chunk.
        return boundary + (2 if text[boundary:boundary + 2] == ". " else 1)

    return target


def chunk_document(
    document: PDFDocument,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split each PDF page into overlapping chunks.

    Chunks never cross page boundaries, so every chunk has an exact
    source page. Chunk indexes start at 1 for each page.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[TextChunk] = []

    for page in document.pages:
        text = normalize_text(page.text)
        if not text:
            continue

        start = 0
        chunk_index = 1

        while start < len(text):
            target = min(start + chunk_size, len(text))
            end = _find_chunk_end(text, start, target)

            if end <= start:
                end = target

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=(
                            f"{document.document_id[:12]}"
                            f"-p{page.page_number}-c{chunk_index}"
                        ),
                        document_id=document.document_id,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        extraction_method=page.extraction_method,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            next_start = max(start + 1, end - overlap)

            # Avoid starting the next chunk in the middle of a word.
            whitespace = text.find(" ", next_start, end)
            if whitespace != -1:
                next_start = whitespace + 1

            start = next_start

    return chunks
