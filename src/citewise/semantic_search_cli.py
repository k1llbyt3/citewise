
"""Search indexed PDF chunks using semantic similarity."""

import argparse
import sys
from pathlib import Path

from citewise.embedding_index import load_embedding_index
from citewise.semantic_retriever import SemanticRetriever


DEFAULT_INDEX_PATH = Path("data/semantic_index.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search PDF chunks using semantic similarity."
    )
    parser.add_argument(
        "query",
        help="Question or phrase to search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display (default: 5).",
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

    if not args.index.is_file():
        print(f"Error: semantic index not found: {args.index}", file=sys.stderr)
        print(
            "Create it first with: uv run python -m citewise.embedding_cli",
            file=sys.stderr,
        )
        return 1

    try:
        index = load_embedding_index(args.index)
        retriever = SemanticRetriever(index)

        results = retriever.search(args.query, top_k=args.top_k)

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No matching passages found.")
        return 0

    print(f"\nSemantic search: {args.query}")
    print(f"Results: {len(results)}\n")

    for rank, result in enumerate(results, start=1):
        filename = result.get("filename", "Unknown file")
        page_number = result["page_number"]
        score = result["score"]

        print(f"{rank}. {filename} — page {page_number}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   {result['text']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
