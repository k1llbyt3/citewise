
"""Search PDF chunks using combined BM25 and semantic rankings."""

import argparse
import sys
from pathlib import Path

from citewise.embedding_index import load_embedding_index
from citewise.hybrid_retriever import HybridRetriever
from citewise.retriever import load_chunks


DEFAULT_REPORTS_DIR = Path("output")
DEFAULT_INDEX_PATH = Path("data/semantic_index.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search PDFs using hybrid keyword and semantic retrieval."
    )
    parser.add_argument("query", help="Question or phrase to search for.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display (default: 5).",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing PDF ingestion reports.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Path to the saved semantic index.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    if not args.reports.is_dir():
        print(
            f"Error: reports directory not found: {args.reports}",
            file=sys.stderr,
        )
        return 1

    if not args.index.is_file():
        print(
            f"Error: semantic index not found: {args.index}",
            file=sys.stderr,
        )
        return 1

    try:
        report_files = sorted(args.reports.glob("*.json"))

        if not report_files:
            raise ValueError(
        f"No JSON ingestion reports found in: {args.reports}"
    )
        chunks = load_chunks(report_files)
        index = load_embedding_index(args.index)

        chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        index_ids = {item["chunk_id"] for item in index["items"]}

        if chunk_ids != index_ids:
            raise ValueError(
                "The reports and semantic index contain different chunk IDs. "
                "Rebuild the embedding index before hybrid searching."
            )

        retriever = HybridRetriever(chunks=chunks, index=index)
        results = retriever.search(args.query, top_k=args.top_k)

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No matching passages found.")
        return 0

    print(f"\nHybrid search: {args.query}")
    print(f"Results: {len(results)}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.get('filename', 'Unknown file')} "
            f"— page {result['page_number']}"
        )
        print(f"   Hybrid score: {result['hybrid_score']:.6f}")
        print(f"   BM25 rank: {result.get('bm25_rank', 'Not ranked')}")
        print(
            f"   Semantic rank: "
            f"{result.get('semantic_rank', 'Not ranked')}"
        )
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   {result['text']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
