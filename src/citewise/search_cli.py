
"""Search processed CiteWise PDF reports from the command line."""

import argparse
from pathlib import Path

from citewise.retriever import BM25Retriever, load_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search processed CiteWise PDF chunks."
    )
    parser.add_argument(
        "query",
        help="Question or keywords to search for.",
    )
    parser.add_argument(
        "--reports",
        default="output",
        help="Directory containing JSON reports (default: output).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of results (default: 5).",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    report_dir = Path(args.reports)

    if not report_dir.is_dir():
        parser.error(f"Report directory does not exist: {report_dir}")

    report_paths = sorted(report_dir.glob("*.json"))

    if not report_paths:
        parser.error(f"No JSON reports found in: {report_dir}")

    try:
        chunks = load_chunks(report_paths)
        retriever = BM25Retriever(chunks)
        results = retriever.search(args.query, top_k=args.top_k)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if not results:
        print("No matching passages found.")
        return 0

    print(f"\nQuery: {args.query}")
    print(f"Matching passages: {len(results)}\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['filename']} — Page {result['page_number']}")
        print(f"   Chunk: {result['chunk_id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
