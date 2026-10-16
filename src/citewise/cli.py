
"""Command-line interface for CiteWise PDF ingestion."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from citewise.ocr import OCRProcessingError
from citewise.pdf_loader import PDFLoadError
from citewise.pipeline import process_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract, OCR, and chunk a PDF with CiteWise."
    )
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file. Defaults to output/<document_id>.json.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk length in characters (default: 1000).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters (default: 150).",
    )
    parser.add_argument(
        "--ocr-language",
        default="eng",
        help="Installed Tesseract language code (default: eng).",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=150,
        help="OCR rendering resolution from 72 to 300 DPI (default: 150).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = process_pdf(
            pdf_path=args.pdf,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            ocr_language=args.ocr_language,
            ocr_dpi=args.ocr_dpi,
        )

        document = result.report.document
        output_path = (
            Path(args.output)
            if args.output
            else Path("output") / f"{document.document_id[:12]}.json"
        )

        payload = {
            "document": asdict(document),
            "chunks": [asdict(chunk) for chunk in result.chunks],
            "pages_still_needing_ocr": (
                result.report.empty_or_low_text_pages
            ),
            "chunk_count": len(result.chunks),
            "status": (
                "needs_review"
                if result.report.empty_or_low_text_pages
                else "success"
            ),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("CiteWise ingestion completed.")
        print(f"Document: {document.filename}")
        print(f"Pages: {document.page_count}")
        print(f"Chunks: {len(result.chunks)}")
        print(f"Pages still needing OCR: {result.report.empty_or_low_text_pages}")
        print(f"JSON report: {output_path}")
        return 0

    except (PDFLoadError, OCRProcessingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
