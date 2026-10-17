
import json

from citewise.hybrid_search_cli import main


def test_cli_reports_missing_reports_directory(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing"
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search_cli",
            "test query",
            "--reports",
            str(missing),
            "--index",
            str(index),
        ],
    )

    assert main() == 1
    assert "reports directory not found" in capsys.readouterr().err


def test_cli_reports_missing_index(tmp_path, monkeypatch, capsys):
    reports = tmp_path / "output"
    reports.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search_cli",
            "test query",
            "--reports",
            str(reports),
            "--index",
            str(tmp_path / "missing.json"),
        ],
    )

    assert main() == 1
    assert "semantic index not found" in capsys.readouterr().err


def test_cli_reports_empty_reports_directory(tmp_path, monkeypatch, capsys):
    reports = tmp_path / "output"
    reports.mkdir()

    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "index_version": 1,
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "items": [],
    }), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "hybrid_search_cli",
            "test query",
            "--reports",
            str(reports),
            "--index",
            str(index),
        ],
    )

    assert main() == 1
    assert "No JSON ingestion reports found" in capsys.readouterr().err
