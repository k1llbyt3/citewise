# CiteWise RAG Evaluation Report: `eval_20261018_155408_retrieval_hybrid`

- **Timestamp**: 20261018_155408
- **Mode**: `retrieval`
- **Retriever**: `hybrid`
- **Live API Execution**: `False`
- **Allow Query Embedding API**: `False`
- **Query Cache**: `evaluation/data/query_cache.json`
- **Dataset**: `evaluation\dataset.json` (hash: `83725f74d6dff8d3`)
- **Cases Evaluated**: 22

## Retrieval Metrics

| Metric | Score |
| :--- | :---: |
| **Mean Recall@1** | 0.4583 |
| **Mean Recall@3** | 0.5729 |
| **Mean Recall@5** | 0.7188 |
| **Mean Reciprocal Rank (MRR)** | 0.6927 |
| **Mean nDCG@5** | 0.6328 |

## Per-Case Breakdown

| Case ID | Category | Expected Ans | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `EVAL-001` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-002` | `direct_factual` | Yes | Rank 1 | R@5=0.67, nDCG=0.70 |
| `EVAL-003` | `direct_factual` | Yes | Rank 4 | R@5=1.00, nDCG=0.43 |
| `EVAL-004` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-005` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=0.88 |
| `EVAL-006` | `paraphrased_semantic` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-007` | `paraphrased_semantic` | Yes | Rank 4 | R@5=0.50, nDCG=0.26 |
| `EVAL-008` | `paraphrased_semantic` | Yes | Rank 3 | R@5=1.00, nDCG=0.50 |
| `EVAL-009` | `multi_passage` | Yes | Rank 4 | R@5=0.33, nDCG=0.20 |
| `EVAL-010` | `multi_passage` | Yes | Rank 1 | R@5=1.00, nDCG=0.92 |
| `EVAL-011` | `multi_page` | Yes | Rank 1 | R@5=0.50, nDCG=0.61 |
| `EVAL-012` | `multi_page` | Yes | Rank 1 | R@5=0.50, nDCG=0.61 |
| `EVAL-013` | `cross_document` | Yes | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-014` | `ocr_scanned` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-015` | `ocr_scanned` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-016` | `semantic_vs_bm25` | Yes | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-017` | `unanswerable` | No | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-018` | `unanswerable` | No | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-019` | `unanswerable` | No | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-020` | `misleading_premise` | No | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-021` | `misleading_premise` | No | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-022` | `ambiguous_insufficient` | No | Miss | R@5=0.00, nDCG=0.00 |
