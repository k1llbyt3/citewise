# CiteWise RAG Evaluation Report: `eval_20261018_162629_answer_hybrid`

- **Timestamp**: 20261018_162629
- **Mode**: `answer`
- **Retriever**: `hybrid`
- **Live API Execution**: `True`
- **Generator Type**: `live_llm`
- **Mock Generation**: `False`
- **Allow Query Embedding API**: `False`
- **Query Cache**: `evaluation/data/query_cache.json`
- **Dataset**: `evaluation\live_test_subset.json` (hash: `b5da54f4514feb9a`)
- **Cases Evaluated**: 3

## Answer & Citation Metrics

| Metric | Score |
| :--- | :---: |
| **Answerability Accuracy** | 1.0000 |
| **Abstention Accuracy** | 1.0000 |
| **Mean Concept Coverage** | 1.0000 |
| **Citation Precision (Chunk)** | 0.7222 |
| **Citation Recall (Chunk)** | 1.0000 |
| **Citation F1 (Chunk)** | 0.8222 |
| **Citation Precision (Page)** | 0.6667 |
| **Citation Recall (Page)** | 1.0000 |
| **Grounded Citation Rate** | 1.0000 |
| **Execution Errors** | 0 |

## Per-Case Breakdown

| Case ID | Category | Expected Ans | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `EVAL-001` | `direct_factual` | Yes | Correct | Coverage=1.00, Prec=0.50 |
| `EVAL-010` | `multi_passage` | Yes | Correct | Coverage=1.00, Prec=0.67 |
| `EVAL-020` | `misleading_premise` | No | Correct | Coverage=1.00, Prec=1.00 |
