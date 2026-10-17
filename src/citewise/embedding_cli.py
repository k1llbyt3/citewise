
"""Build a semantic index from existing CiteWise JSON reports."""

import argparse
from pathlib import Path

from citewise.embedding_index import (
    build_embedding_index,
    save_embedding_index,
)
from citewise.retriever import load_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local semantic index from CiteWise reports."
    )
    parser.add_argument(
        "--reports",
        default="output",
        help="Directory containing processed JSON reports.",
    )
    parser.add_argument(
        "--index",
        default="data/semantic_index.json",
        help="Where to save the generated embedding index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks per embedding request.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    report_dir = Path(args.reports)

    if not report_dir.is_dir():
        parser.error(f"Report directory does not exist: {report_dir}")

    report_paths = sorted(report_dir.glob("*.json"))

    if not report_paths:
        parser.error(f"No JSON reports found in: {report_dir}")

    try:
        chunks = load_chunks(report_paths)

        if not chunks:
            parser.error("The reports contain no chunks to embed.")

        index = build_embedding_index(
            chunks,
            batch_size=args.batch_size,
        )
        save_embedding_index(index, args.index)

    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Indexed chunks: {len(index['items'])}")
    print(f"Embedding model: {index['model']}")
    print(f"Vector dimensions: {index['dimensions']}")
    print(f"Saved index: {args.index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
