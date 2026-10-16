
"""Automated tests for the CiteWise command-line interface."""

import json
import sys

import pymupdf

from citewise.cli import main


def create_text_pdf(path):
    """Create a small PDF with selectable text."""
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "CiteWise CLI test document with readable text for extraction.",
            fontsize=12,
        )
        pdf.save(path)


def test_cli_creates_valid_json_report(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "sample.pdf"
    output_path = tmp_path / "report.json"
    create_text_pdf(pdf_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["citewise", str(pdf_path), "--output", str(output_path)],
    )

    exit_code = main()

    assert exit_code == 0
    assert output_path.exists()

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["status"] == "success"
    assert report["document"]["filename"] == "sample.pdf"
    assert report["document"]["page_count"] == 1
    assert report["chunk_count"] >= 1
    assert len(report["chunks"]) == report["chunk_count"]
    assert report["pages_still_needing_ocr"] == []

    chunk = report["chunks"][0]
    assert chunk["document_id"] == report["document"]["document_id"]
    assert chunk["page_number"] == 1
    assert chunk["text"]
    assert "CiteWise CLI test document" in chunk["text"]

    assert "CiteWise ingestion completed." in capsys.readouterr().out


def test_cli_uses_default_output_path(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    create_text_pdf(pdf_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["citewise", str(pdf_path)])

    exit_code = main()

    assert exit_code == 0
    output_files = list((tmp_path / "output").glob("*.json"))
    assert len(output_files) == 1

    report = json.loads(output_files[0].read_text(encoding="utf-8"))
    assert report["document"]["filename"] == "sample.pdf"


def test_cli_handles_missing_pdf(tmp_path, monkeypatch, capsys):
    missing_pdf = tmp_path / "missing.pdf"
    output_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "citewise",
            str(missing_pdf),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()

    assert exit_code == 1
    assert not output_path.exists()
    assert "Error:" in capsys.readouterr().err
