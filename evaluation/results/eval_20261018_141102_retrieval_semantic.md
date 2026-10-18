# CiteWise RAG Evaluation Report: `eval_20261018_141102_retrieval_semantic`

- **Timestamp**: 20261018_141102
- **Mode**: `retrieval`
- **Retriever**: `semantic`
- **Live API Execution**: `False`
- **Dataset**: `evaluation\dataset.json` (hash: `83725f74d6dff8d3`)
- **Cases Evaluated**: 22

## Retrieval Metrics

| Metric | Score |
| :--- | :---: |
| **Mean Recall@1** | 0.3646 |
| **Mean Recall@3** | 0.5417 |
| **Mean Recall@5** | 0.6250 |
| **Mean Reciprocal Rank (MRR)** | 0.5938 |
| **Mean nDCG@5** | 0.5598 |

## Per-Case Breakdown

| Case ID | Category | Expected Ans | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `EVAL-001` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-002` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=0.97 |
| `EVAL-003` | `direct_factual` | Yes | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-004` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=1.00 |
| `EVAL-005` | `direct_factual` | Yes | Rank 1 | R@5=1.00, nDCG=0.88 |
| `EVAL-006` | `paraphrased_semantic` | Yes | Rank 2 | R@5=1.00, nDCG=0.63 |
| `EVAL-007` | `paraphrased_semantic` | Yes | Rank 1 | R@5=1.00, nDCG=0.85 |
| `EVAL-008` | `paraphrased_semantic` | Yes | Rank 2 | R@5=1.00, nDCG=0.63 |
| `EVAL-009` | `multi_passage` | Yes | Miss | R@5=0.00, nDCG=0.00 |
| `EVAL-010` | `multi_passage` | Yes | Rank 2 | R@5=0.50, nDCG=0.39 |
| `EVAL-011` | `multi_page` | Yes | Miss | R@5=0.00, nDCG=0.00 |
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
