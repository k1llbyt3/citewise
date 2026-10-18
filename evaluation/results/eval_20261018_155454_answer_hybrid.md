# CiteWise RAG Evaluation Report: `eval_20261018_155454_answer_hybrid`

- **Timestamp**: 20261018_155454
- **Mode**: `answer`
- **Retriever**: `hybrid`
- **Live API Execution**: `False`
- **Allow Query Embedding API**: `False`
- **Query Cache**: `evaluation/data/query_cache.json`
- **Dataset**: `evaluation\dataset.json` (hash: `83725f74d6dff8d3`)
- **Cases Evaluated**: 22

## Answer & Citation Metrics

| Metric | Score |
| :--- | :---: |
| **Answerability Accuracy** | 0.7273 |
| **Abstention Accuracy** | 0.0000 |
| **Mean Concept Coverage** | 0.1515 |
| **Citation Precision (Chunk)** | 0.4545 |
| **Citation Recall (Chunk)** | 0.3333 |
| **Citation F1 (Chunk)** | 0.3712 |
| **Citation Precision (Page)** | 0.5000 |
| **Citation Recall (Page)** | 0.3788 |
| **Grounded Citation Rate** | 1.0000 |
| **Execution Errors** | 0 |

## Per-Case Breakdown

| Case ID | Category | Expected Ans | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `EVAL-001` | `direct_factual` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-002` | `direct_factual` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-003` | `direct_factual` | Yes | Correct | Coverage=0.00, Prec=0.00 |
| `EVAL-004` | `direct_factual` | Yes | Correct | Coverage=1.00, Prec=1.00 |
| `EVAL-005` | `direct_factual` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-006` | `paraphrased_semantic` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-007` | `paraphrased_semantic` | Yes | Correct | Coverage=0.33, Prec=0.00 |
| `EVAL-008` | `paraphrased_semantic` | Yes | Correct | Coverage=0.00, Prec=0.00 |
| `EVAL-009` | `multi_passage` | Yes | Correct | Coverage=0.00, Prec=0.00 |
| `EVAL-010` | `multi_passage` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-011` | `multi_page` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-012` | `multi_page` | Yes | Correct | Coverage=0.00, Prec=1.00 |
| `EVAL-013` | `cross_document` | Yes | Correct | Coverage=0.00, Prec=0.00 |
| `EVAL-014` | `ocr_scanned` | Yes | Correct | Coverage=1.00, Prec=1.00 |
| `EVAL-015` | `ocr_scanned` | Yes | Correct | Coverage=1.00, Prec=1.00 |
| `EVAL-016` | `semantic_vs_bm25` | Yes | Correct | Coverage=0.00, Prec=0.00 |
| `EVAL-017` | `unanswerable` | No | Incorrect | Coverage=0.00, Prec=0.00 |
| `EVAL-018` | `unanswerable` | No | Incorrect | Coverage=0.00, Prec=0.00 |
| `EVAL-019` | `unanswerable` | No | Incorrect | Coverage=0.00, Prec=0.00 |
| `EVAL-020` | `misleading_premise` | No | Incorrect | Coverage=0.00, Prec=0.00 |
| `EVAL-021` | `misleading_premise` | No | Incorrect | Coverage=0.00, Prec=0.00 |
| `EVAL-022` | `ambiguous_insufficient` | No | Incorrect | Coverage=0.00, Prec=0.00 |
