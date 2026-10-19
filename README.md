# CiteWise

CiteWise is a lightweight, local, page-aware Retrieval-Augmented Generation (RAG) question-answering system for PDF documents. It ingests digital and scanned PDFs, normalizes and chunks text while maintaining full document provenance, indexes passages using BM25 keyword matching and dense vector embeddings, retrieves relevant context using Reciprocal Rank Fusion (RRF), validates passage sufficiency using a local evidence gate, and generates strictly grounded answers with verifiable PDF citations.

---

## Architecture Overview

```text
       [ PDF Document ] (Digital or Scanned)
              │
              ▼
   ┌──────────────────────────────────────────┐
   │ Ingestion Pipeline (PyMuPDF Loader)      │
   │ ├─ Extracts native text                  │
   │ └─ Low-text/Scanned fallback: Tesseract  │
   └──────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────┐
   │ Chunker (Page-Aware Provenance)          │
   │ ├─ Preserves paragraph structures        │
   │ ├─ Normalizes whitespace & unicode       │
   │ └─ Default: 1000 chars, 150 overlap      │
   └──────────────────────────────────────────┘
              │
              ├───────────────────────────────┐
              ▼                               ▼
   ┌──────────────────────────┐   ┌──────────────────────────┐
   │ BM25 Keyword Index       │   │ Dense Embedding Index    │
   │ ├─ Term frequencies      │   │ ├─ gemini-embedding-001  │
   │ ├─ Morphological bounds  │   │ ├─ 768 dimensions        │
   │ └─ Pure Python           │   │ └─ Cosine similarity     │
   └──────────────────────────┘   └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
   ┌──────────────────────────────────────────┐
   │ Hybrid Retrieval: Reciprocal Rank Fusion │
   │ └─ RRF Score: Σ 1 / (60 + rank_i)        │
   └──────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────┐
   │ Pre-Generation Evidence Gate             │
   │ ├─ Lexical overlap analysis              │
   │ ├─ Multi-passage combined detection      │
   │ ├─ Score separation & threshold checks   │
   │ └─ Rejects out-of-context queries early  │
   └──────────────────────────────────────────┘
          │ (Pass / Borderline)        │ (Reject: 0 live API calls)
          ▼                            ▼
   ┌──────────────────────────┐   ┌──────────────────────────┐
   │ Grounded Answer Gen      │   │ Truthful Abstention      │
   │ ├─ Gemini 2.5 Flash      │   │ "I couldn't find enough  │
   │ ├─ Structured schema     │   │ information in the       │
   │ ├─ Bracketed citations   │   │ provided PDF passages..."│
   │ └─ Verifiable sources    │   └──────────────────────────┘
   └──────────────────────────┘
```

---

## Key Features

- **Dual-Engine Ingestion**: Reads native digital text with PyMuPDF and automatically triggers Tesseract OCR when a page contains fewer than 20 characters.
- **Page-Aware Provenance**: Every chunk preserves its original filename, document ID hash, and source page number so citations can always be traced directly to the original document.
- **Hybrid Retrieval (RRF)**: Combines pure Python BM25 keyword relevance with 768-dimensional dense semantic embeddings using Reciprocal Rank Fusion ($k=60$).
- **Pre-Generation Evidence Gate**: Validates retrieved passages locally before calling external LLMs. Rejects irrelevant queries immediately, conserving API quota and preventing hallucinations.
- **Truthful Abstention Protocol**: When evidence is absent, conflicting, or out-of-context, CiteWise abstains with a canonical refusal message rather than guessing.
- **Zero-Dependency Web UI**: Runs locally via Python's built-in `http.server` with PDF drag-and-drop ingestion, corpus selection, math notation rendering ($\LaTeX$ via KaTeX with offline typography fallback), code block highlighting, and collapsible transparency sections for rejected queries.
- **Evaluation & Query Cache Framework**: Automated benchmarking suite supporting Recall@K, Mean Reciprocal Rank (MRR), nDCG@5, concept coverage, answerability, and citation precision/recall/F1 metrics.

---

## System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python**: Python 3.11 or newer.
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended for speed and RAM efficiency).
- **OCR Engine (Optional)**: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for scanned PDFs (default Windows path: `C:\Program Files\Tesseract-OCR\tessdata`).
- **Gemini API Key (Optional)**: Required only for semantic embeddings and live answer generation. Ingestion, BM25 retrieval, the evidence gate, and mock evaluation run 100% offline.

---

## Installation & Setup

### 1. Clone Repository & Install Dependencies

Using `uv` (recommended):
```powershell
# Windows PowerShell
uv sync
```
Or on Linux / macOS:
```bash
uv sync
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:
```powershell
# Windows PowerShell
Copy-Item .env.example .env

# Windows Command Prompt
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` with your settings:
```ini
# Required for semantic search and live answer generation
GEMINI_API_KEY="your-gemini-api-key-here"

# Process-local rate pacing between Gemini API requests (seconds)
# Set to 21.0 or higher for free-tier Gemini API accounts (15 RPM limit)
CITEWISE_MIN_API_INTERVAL_SECONDS="21.0"
```

---

## Quickstart Tutorial

### Step 1: Ingest Sample PDFs

Ingest digital and scanned PDF documents into structured JSON chunk reports:

```powershell
# Ingest digital PDF
uv run python -m citewise.cli samples/text_sample.pdf --output output/text_report.json

# Ingest scanned PDF with automatic OCR fallback
uv run python -m citewise.cli samples/scan_sample.pdf --output output/scan_report.json
```

### Step 2: Build the Semantic Index (Optional)

Generate dense embeddings for the ingested chunks:

```powershell
uv run python -m citewise.embedding_cli --reports output --index data/semantic_index.json
```

### Step 3: Search Documents

#### BM25 Keyword Search (100% Offline)
```powershell
uv run python -m citewise.search_cli "chunking" --reports output --top-k 5
```

#### Dense Semantic Search
```powershell
uv run python -m citewise.semantic_search_cli "Why is chunking useful?" --index data/semantic_index.json --top-k 5
```

#### Hybrid Search (BM25 + Semantic via RRF)
```powershell
uv run python -m citewise.hybrid_search_cli "Why does chunking matter?" --reports output --index data/semantic_index.json --top-k 5
```

### Step 4: Ask Questions with Verifiable Citations

```powershell
uv run python -m citewise.answer_cli "What is the default chunk size of CiteWise?" --reports output --index data/semantic_index.json
```

Example Output:
```text
Question:
What is the default chunk size of CiteWise?

Answer:
The default chunk size of CiteWise is 1000 characters with an overlap of 150 characters.

Sources:
- text_sample.pdf, page 3 (chunk: b637c456e6e5-p3-c1)
```

---

## Local Web Interface

CiteWise includes a standalone local web console with zero npm/pip UI dependencies:

```powershell
uv run python -m citewise.web_ui --host 127.0.0.1 --port 8080 --reports user_data/reports --uploads user_data/uploads
```

Once started, open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

### Web UI Features

1. **PDF Upload & Ingestion**: Drag and drop PDF files into the browser. Ingestion reports are automatically saved under `user_data/reports/`.
2. **Corpus Management**: View ingested documents, page counts, chunk counts, and select/deselect specific documents to target question answering.
3. **Dual Retriever Modes**: Select between BM25 (fully offline) or Hybrid retrieval (BM25 + Semantic embeddings) with selectable Top-K limits (3, 5, or 8 passages).
4. **Interactive Citations**: Clickable source tags (`[S1]`, `[S2]`) smoothly scroll to and highlight the corresponding source card.
5. **Expandable Source Excerpts**: Cards display concise 240-character snippets by default, with a "Show full excerpt" button to inspect the complete context.
6. **Mathematical Formatting**: Renders LaTeX expressions (`\(...\)`, `\[...\]`, `$...$`, `$$...$$`) using KaTeX when online, with an offline typography fallback for mathematical symbols.
7. **Code Block Highlighting**: Formats code blocks with language indicators and clean typography.
8. **Collapsible Transparency Section**: When a question cannot be answered from the document corpus, retrieved passages are hidden inside a collapsed `<details>` section (`Retrieved passages (not sufficient to answer)`) rather than misleadingly presented as supporting evidence.

---

## Evaluation & Benchmarking

CiteWise includes an automated evaluation harness for quantitative benchmarking against gold-standard datasets.

### 1. Offline Retrieval Benchmark

Evaluate retrieval performance without API calls using the pre-cached query vectors:

```powershell
# BM25 Keyword Benchmark
uv run python -m citewise.evaluate_cli --mode retrieval --retriever bm25

# Semantic Retrieval Benchmark (runs offline against query cache)
uv run python -m citewise.evaluate_cli --mode retrieval --retriever semantic

# Hybrid Retrieval Benchmark (RRF)
uv run python -m citewise.evaluate_cli --mode retrieval --retriever hybrid
```

### 2. Full Pipeline Evaluation with Offline Mock Responses

Verify retrieval, answerability, abstention validation, and citation matching with zero API credits:

```powershell
# Standard 22-case dataset
uv run python -m citewise.evaluate_cli --mode full --retriever bm25 --dataset evaluation/dataset.json --mock-responses tests/fixtures/sample_answer_fixtures.json

# Advanced multi-document dataset
uv run python -m citewise.evaluate_cli --mode full --retriever bm25 --dataset evaluation/advanced_dataset.json --mock-responses tests/fixtures/advanced_answer_fixtures.json
```

### 3. Rate-Safe Live Evaluation

Live evaluations against the Gemini API enforce rate pacing and a strict 3-case cap:

```powershell
uv run python -m citewise.evaluate_cli --mode answer --retriever bm25 --dataset evaluation/live_test_subset.json --live --max-cases 3
```

Benchmark reports are automatically saved to `evaluation/results/` in both JSON and Markdown formats.

---

## Testing

Run the full automated test suite with `pytest`:

```powershell
# Run all tests (fast & quiet)
uv run pytest -q

# Run specific test modules
uv run pytest tests/test_evidence_gate.py -v
uv run pytest tests/test_ui_contract.py -v
uv run pytest tests/test_retriever.py -v
```

All 250 unit, harness, and contract tests execute offline without external network or API dependencies.

---

## Project Structure

```text
citewise/
├── data/                         # Local semantic embedding indexes
├── evaluation/                   # Benchmarking suite & datasets
│   ├── advanced_dataset.json     # Advanced multi-document evaluation dataset
│   ├── dataset.json              # Standard 22-case gold evaluation dataset
│   ├── live_test_subset.json     # 3-case live test subset
│   ├── data/                     # Semantic index & query embedding cache
│   ├── reports/                  # Benchmark corpus ingestion reports
│   └── results/                  # Generated benchmark evaluation reports
├── output/                       # Default CLI ingestion reports
├── samples/                      # Sample text and scanned PDFs
├── src/citewise/
│   ├── answer_cli.py             # Question answering CLI
│   ├── answer_eval.py            # Evaluation scoring metrics
│   ├── answer_generator.py       # Grounded generation with citation enforcement
│   ├── api_limits.py             # Process-local rate pacing safeguard
│   ├── chunker.py                # Page-aware text normalization and chunking
│   ├── cli.py                    # PDF ingestion CLI
│   ├── config.py                 # Configuration constants & defaults
│   ├── embedding_cli.py          # Semantic index builder CLI
│   ├── embedding_index.py        # Local vector index storage & cosine similarity
│   ├── embeddings.py             # Gemini embedding API integration
│   ├── eval_dataset.py           # Evaluation dataset schemas & validators
│   ├── evaluate_cli.py           # Benchmark evaluation CLI
│   ├── evidence_gate.py          # Pre-generation lexical evidence gating
│   ├── hybrid_retriever.py       # Reciprocal Rank Fusion retriever
│   ├── hybrid_search_cli.py      # Hybrid search CLI
│   ├── models.py                 # Core domain dataclasses
│   ├── ocr.py                    # Tesseract OCR fallback engine
│   ├── pdf_loader.py             # PyMuPDF document extraction
│   ├── pipeline.py               # Ingestion pipeline coordinator
│   ├── query_cache.py            # Query embedding cache layer
│   ├── retrieval_eval.py         # Retrieval metrics (Recall@K, MRR, nDCG)
│   ├── retriever.py              # Pure Python BM25 keyword retriever
│   ├── search_cli.py             # BM25 search CLI
│   ├── semantic_retriever.py     # Dense vector retriever
│   ├── semantic_search_cli.py    # Vector search CLI
│   └── web_ui.py                 # Standalone local HTTP web interface
├── tests/                        # Offline pytest test suite (250 tests)
│   └── fixtures/                 # Evaluation test fixtures and mock responses
├── user_data/                    # Web UI isolated user document workspace
│   ├── reports/                  # User ingested JSON chunk reports
│   └── uploads/                  # Uploaded user PDF files
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # Documentation and usage guide
```

---

## Troubleshooting & FAQ

### 1. `TesseractNotFoundError` or OCR Warnings on Scanned PDFs
- **Cause**: Tesseract OCR is not installed or not in the expected path.
- **Resolution**: Install Tesseract OCR from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). On Windows, the default installation path is `C:\Program Files\Tesseract-OCR\tessdata`. If installed elsewhere, ensure `TESSDATA_PREFIX` is set in your environment. Digital PDFs with native text layers do not require Tesseract.

### 2. HTTP 429 Quota Exceeded / Rate Limits
- **Cause**: Gemini API free-tier accounts are limited to 15 requests per minute.
- **Resolution**: CiteWise includes a built-in process-local rate pacer (`CITEWISE_MIN_API_INTERVAL_SECONDS`). Ensure this is set to at least `21.0` seconds in your `.env` file to stay under the rate limit.

### 3. "The reports and semantic index contain different chunk IDs"
- **Cause**: `hybrid_search_cli.py` or `answer_cli.py` was invoked with a reports directory whose chunks do not match the specified index file.
- **Resolution**: Use matching pairs of reports and index files. For default sample data:
  `--reports output --index data/semantic_index.json`
  For evaluation data:
  `--reports evaluation/reports --index evaluation/data/semantic_index.json`

### 4. Running 100% Offline
- All ingestion, chunking, BM25 searching, evidence gating, and evaluation with mock responses can run completely offline without an internet connection or Gemini API key. Simply omit `--live` and use `--retriever bm25`.

---

## License

This project is licensed under the MIT License.
