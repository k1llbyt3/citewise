# CiteWise

CiteWise is a Python-based PDF ingestion pipeline for a retrieval-augmented generation (RAG) system. It extracts text from PDFs, uses OCR for pages with little or no extractable text, splits the extracted content into chunks, and saves the results as JSON.

The current implementation covers document ingestion only. It does not yet include embeddings, semantic search, retrieval, or question answering.

## Features

- Extracts selectable text from PDF files using PyMuPDF.
- Detects pages with fewer than 20 extracted characters and flags them for OCR.
- Uses Tesseract OCR for scanned or low-text pages.
- Normalizes whitespace while preserving paragraph boundaries.
- Splits text into chunks while retaining document IDs, page numbers, chunk IDs, and extraction methods.
- Saves document metadata, page text, chunks, and processing status to JSON.
- Validates PDF size, page count, file type, and basic input errors.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Tesseract OCR with English language data (`eng`)

The current PDF loader limits inputs to 25 MB and 300 pages.

## Setup

Clone or open the project, then run these commands from the project root:

```cmd
uv sync
```

Install Tesseract OCR separately on Windows if it is not already installed. The current OCR implementation expects its language data at:

```text
C:\Program Files\Tesseract-OCR\tessdata
```

If Tesseract is installed elsewhere, update the `tessdata` path in `src/citewise/ocr.py` to match your installation.

## Run the pipeline

Process a text-based PDF:

```cmd
uv run python -m citewise.cli samples\text_sample.pdf
```

Process a scanned PDF:

```cmd
uv run python -m citewise.cli samples\scan_sample.pdf
```

Specify an output file:

```cmd
uv run python -m citewise.cli samples\text_sample.pdf --output output\text_report.json
```

Change the chunk size and overlap:

```cmd
uv run python -m citewise.cli samples\text_sample.pdf --chunk-size 800 --overlap 100
```

The default chunk size is 1,000 characters, with an overlap of 150 characters.

When no output path is specified, the CLI writes a JSON file under `output/`, using the document ID as its filename.

## JSON output

The report contains:

- `document`: document ID, filename, page count, file size, and extracted page records.
- `chunks`: chunk text and its source metadata.
- `pages_still_needing_ocr`: pages that remain flagged after OCR processing.
- `chunk_count`: number of chunks generated.
- `status`: `success` when no pages remain flagged, or `needs_review` otherwise.

Each page record includes its page number, extracted text, extraction method, and OCR flag. Each chunk includes its ID, document ID, page number, chunk index, text, and extraction method.

A `success` status indicates that no pages remain flagged for OCR; it does not guarantee perfect OCR accuracy or correct interpretation of the document.

## Tests

Run the complete test suite:

```cmd
uv run pytest -v
```

Run tests for an individual component:

```cmd
uv run pytest tests\test_pdf_loader.py -v
uv run pytest tests\test_ocr.py -v
uv run pytest tests\test_chunker.py -v
uv run pytest tests\test_pipeline.py -v
uv run pytest tests\test_cli.py -v
```

The tests cover PDF validation, text extraction, OCR fallback behavior, chunking, pipeline integration, and CLI output.

## Project structure

```text
citewise/
├── samples/                 # Sample text and scanned PDFs
├── output/                  # Generated JSON reports
├── src/
│   └── citewise/
│       ├── chunker.py       # Text normalization and chunking
│       ├── cli.py           # Command-line interface and JSON output
│       ├── config.py        # Processing limits
│       ├── models.py        # Document, page, report, and chunk models
│       ├── ocr.py           # Tesseract OCR fallback
│       ├── pdf_loader.py    # PDF validation and text extraction
│       └── pipeline.py      # Ingestion workflow
├── tests/                   # Automated tests
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency versions
└── README.md
```

## Current limitations

- OCR is configured for English by default.
- OCR accuracy depends on scan quality, orientation, layout, and language data.
- Chunks are created independently for each page.
- Chunk sizes are measured in characters, not tokens.
- The OCR data directory is currently configured with a Windows-specific path.
- The pipeline does not yet generate embeddings or support retrieval and question answering.
