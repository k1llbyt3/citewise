
"""Answer PDF questions using hybrid retrieval and grounded generation."""

import argparse
import sys
from pathlib import Path

from citewise.answer_generator import generate_answer
from citewise.embedding_index import load_embedding_index
from citewise.hybrid_retriever import HybridRetriever
from citewise.retriever import load_chunks


DEFAULT_REPORTS_DIR = Path("output")
DEFAULT_INDEX_PATH = Path("data/semantic_index.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask questions about PDFs and receive cited answers."
    )
    parser.add_argument("question", help="Question to answer from your PDFs.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve (default: 5).",
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

    if not args.question.strip():
        parser.error("Question must not be empty.")

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

        report_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
        index_chunk_ids = {item["chunk_id"] for item in index["items"]}

        if report_chunk_ids != index_chunk_ids:
            raise ValueError(
                "The reports and semantic index contain different chunk IDs. "
                "Rebuild the embedding index before answering questions."
            )

        retriever = HybridRetriever(chunks=chunks, index=index)
        passages = retriever.search(args.question, top_k=args.top_k)

        result = generate_answer(args.question, passages)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nQuestion:")
    print(args.question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    if not result["citations"]:
        print("No supporting PDF citations available.")
    else:
        for citation in result["citations"]:
            print(
                f"- {citation['filename']}, "
                f"page {citation['page_number']} "
                f"(chunk: {citation['chunk_id']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
