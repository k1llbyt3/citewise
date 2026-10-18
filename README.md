# CiteWise

CiteWise is a local, page-aware Retrieval-Augmented Generation (RAG) question-answering pipeline for PDF documents. It extracts clean text, automatically applies OCR fallback to scanned pages, indexes chunks with keyword and vector embeddings, retrieves relevant passages using Reciprocal Rank Fusion (RRF), and generates strictly grounded answers with verifiable PDF citations.

---

## Key Features

- **Dual-Engine Ingestion**: Extracts digital text via PyMuPDF; automatically falls back to Tesseract OCR when extractable characters fall below threshold (<20 chars).
- **Page-Aware Chunking & Provenance**: Normalizes whitespace and creates character chunks while preserving paragraph boundaries, document IDs, page numbers, and chunk indices.
- **Hybrid Retrieval (RRF)**: Combines local BM25 keyword rankings with Gemini 768-dimensional semantic embeddings (`gemini-embedding-001`) via Reciprocal Rank Fusion (`HybridRetriever`).
- **Grounded Answer Generation**: Uses `gemini-2.5-flash` with strict evidence prompts and structured JSON schemas to eliminate hallucinations.
- **Accurate Citation Mapping**: Resolves cited source IDs back to original filenames, page numbers, and chunk IDs.
- **Truthful Abstention**: Canonical fallback refusal (*"I couldn't find enough information in the provided PDF passages to answer this question reliably."*) on unanswerable or misleading-premise questions.
- **Persistent Query Embedding Cache**: Caches query vectors locally to enable 100% offline retrieval benchmarks without unexpected API costs.
- **API Safeguards & Rate Pacing**: Enforces rate pacing (`CITEWISE_MIN_API_INTERVAL_SECONDS`) and strictly caps live evaluation runs at $\le 3$ cases.
- **Comprehensive Evaluation Harness**: 22-case gold-standard evaluation dataset across 8 question categories with automated metric reporting (Recall@K, MRR, nDCG@5, Answerability Accuracy, Abstention Accuracy, Concept Coverage, and Citation Precision/Recall/F1).

---

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Tesseract OCR (Windows default path: `C:\Program Files\Tesseract-OCR\tessdata`)
- Google Gemini API Key (for semantic indexing and live generation)

---

## Installation & Setup

1. **Clone or open the repository**, then sync dependencies:
   ```cmd
   uv sync
   ```

2. **Configure Environment Variables**:
   Copy the example environment file:
   ```cmd
   copy .env.example .env
   ```
   Edit `.env` to supply your API key and optional rate-limit pacing:
   ```ini
   GEMINI_API_KEY="your-gemini-api-key-here"
   CITEWISE_MIN_API_INTERVAL_SECONDS="1.0"
   ```
   *(Note: Set `CITEWISE_MIN_API_INTERVAL_SECONDS="21"` if operating under free-tier Gemini rate limits).*

3. **Install Tesseract OCR**:
   Install Tesseract OCR if scanning scanned PDFs. Ensure English language data (`eng.traineddata`) is installed in `C:\Program Files\Tesseract-OCR\tessdata`, or adjust the path in `src/citewise/ocr.py`.

---

## Step-by-Step Usage Guide

### 1. Ingest PDF Documents
Process text-based or scanned PDFs into structured chunk reports:

```cmd
# Process text PDF
uv run python -m citewise.cli samples\text_sample.pdf --output output\text_report.json

# Process scanned PDF with OCR
uv run python -m citewise.cli samples\scan_sample.pdf --output output\scan_report.json
```

Default parameters: `--chunk-size 1000 --overlap 150`.

### 2. Build the Semantic Index (API required)
Generate 768-dimensional embeddings for all extracted chunks and save them locally:

```cmd
uv run python -m citewise.embedding_cli --reports output --index data/semantic_index.json
```

### 3. Search Processed Documents

#### A. BM25 Keyword Search (100% Offline)
```cmd
uv run python -m citewise.search_cli "chunking" --reports output --top-k 5
```

#### B. Semantic Vector Search
```cmd
uv run python -m citewise.semantic_search_cli "Why is chunking useful?" --index data/semantic_index.json --top-k 5
```

#### C. Hybrid Retrieval (BM25 + Semantic via RRF)
```cmd
uv run python -m citewise.hybrid_search_cli "Why does chunking matter?" --reports output --index data/semantic_index.json --top-k 5
```

### 4. Ask Questions with Grounded PDF Citations
Ask questions and receive answers cited down to filename and page number:

```cmd
# Run against the sample corpus
uv run python -m citewise.answer_cli "Why is chunking useful for RAG?" --reports output --index data/semantic_index.json

# Run against the evaluation corpus
uv run python -m citewise.answer_cli "What are the four primary functions of the NIST AI RMF Core?" --reports evaluation/reports --index evaluation/data/semantic_index.json
```

---

## Evaluation & Benchmarking

CiteWise includes a 22-case gold-standard evaluation dataset covering direct factual questions, paraphrased queries, multi-passage synthesis, cross-document comparison, OCR scanned pages, and unanswerable/misleading queries.

### 1. Offline Retrieval Benchmark (Zero API calls)
Evaluate BM25, Semantic, or Hybrid retrieval using the local query cache:

```cmd
# BM25 Keyword Benchmark
uv run python -m citewise.evaluate_cli --mode retrieval --retriever bm25

# Semantic Retrieval Benchmark (reads cached vectors)
uv run python -m citewise.evaluate_cli --mode retrieval --retriever semantic

# Hybrid Retrieval Benchmark (RRF)
uv run python -m citewise.evaluate_cli --mode retrieval --retriever hybrid
```

### 2. Offline Answer Evaluation with Test Fixtures
Evaluate answer evaluation scoring, abstention logic, and citation metrics completely offline:

```cmd
uv run python -m citewise.evaluate_cli --mode answer --retriever hybrid --mock-responses tests/fixtures/sample_answer_fixtures.json --max-cases 3
```

### 3. Query Cache Population (Opt-in API call)
Populate missing query embeddings into `evaluation/data/query_cache.json` before running offline benchmarks:

```cmd
uv run python -m citewise.evaluate_cli --mode retrieval --retriever semantic --allow-query-embedding-api
```

### 4. Controlled Live Answer-Quality Test
Execute live generation against Gemini LLM (strictly capped at $\le 3$ cases by CLI guardrails):

```cmd
uv run python -m citewise.evaluate_cli --mode answer --retriever hybrid --dataset evaluation/live_test_subset.json --live --max-cases 3
```

Evaluation results are written to `evaluation/results/eval_<timestamp>_<mode>_<retriever>.json` and `.md` with detailed case inspection breakdowns.

---

## Automated Tests

Run the complete test suite:

```cmd
uv run pytest -v
```

All 166 unit, harness, and end-to-end CLI tests run offline with zero external network or API calls.

---

## Project Structure

```text
citewise/
├── samples/                 # Sample text, scanned PDFs, and evaluation corpus
├── output/                  # Ingestion reports (JSON)
├── data/                    # Local semantic embedding indices
├── evaluation/              # Evaluation dataset, query cache, corpus reports, and results
│   ├── dataset.json         # 22-case gold evaluation dataset
│   ├── live_test_subset.json# 3-case pilot subset
│   ├── data/                # Evaluation semantic index & query_cache.json
│   ├── reports/             # 4 evaluation document ingestion reports (241 chunks)
│   └── results/             # Timestamped JSON and Markdown benchmark reports
├── src/
│   └── citewise/
│       ├── answer_cli.py    # Interactive Q&A CLI with citations
│       ├── answer_eval.py   # Deterministic answerability & citation metric engine
│       ├── answer_generator.py # Grounded Gemini answer generation & prompt defense
│       ├── api_limits.py    # Process-local API rate pacing
│       ├── chunker.py       # Page-aware whitespace normalization & text chunker
│       ├── cli.py           # Document ingestion CLI
│       ├── embedding_cli.py # Batch semantic index creation CLI
│       ├── embedding_index.py # Local JSON index persistence & validation
│       ├── embeddings.py    # Gemini embedding API interface
│       ├── eval_dataset.py  # Dataset schema validation & models
│       ├── evaluate_cli.py  # Comprehensive evaluation CLI harness
│       ├── hybrid_retriever.py # Reciprocal Rank Fusion (BM25 + Semantic)
│       ├── hybrid_search_cli.py # Hybrid search CLI
│       ├── models.py        # Core dataclass models
│       ├── ocr.py           # Tesseract OCR fallback engine
│       ├── pdf_loader.py    # PyMuPDF extraction and input validation
│       ├── pipeline.py      # End-to-end document ingestion coordinator
│       ├── query_cache.py   # Persistent query-embedding cache
│       ├── retrieval_eval.py# Information retrieval metrics (Recall@K, MRR, nDCG)
│       ├── retriever.py     # Pure Python BM25 keyword retriever
│       ├── search_cli.py    # BM25 search CLI
│       ├── semantic_retriever.py # Cosine-similarity vector retriever
│       └── semantic_search_cli.py # Semantic search CLI
├── tests/                   # Automated pytest suite (166 tests)
├── pyproject.toml           # Project metadata and dependencies
└── README.md
```
