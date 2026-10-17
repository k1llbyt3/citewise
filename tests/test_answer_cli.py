
import json

from citewise import answer_cli


def setup_inputs(tmp_path):
    reports = tmp_path / "output"
    reports.mkdir()

    # load_chunks is mocked in these tests, so the report can be empty JSON.
    (reports / "report.json").write_text("{}", encoding="utf-8")

    index = tmp_path / "index.json"
    index.write_text(
        json.dumps({
            "index_version": 1,
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "items": [
                {"chunk_id": "chunk-1"}
            ],
        }),
        encoding="utf-8",
    )

    return reports, index


def configure_mocks(monkeypatch, passages, answer_result):
    monkeypatch.setattr(
        answer_cli,
        "load_chunks",
        lambda paths: [
            {
                "chunk_id": "chunk-1",
                "filename": "guide.pdf",
                "page_number": 2,
                "text": "Chunking divides documents into smaller passages.",
            }
        ],
    )

    monkeypatch.setattr(
        answer_cli,
        "load_embedding_index",
        lambda path: {
            "index_version": 1,
            "model": "gemini-embedding-001",
            "dimensions": 768,
            "items": [{"chunk_id": "chunk-1"}],
        },
    )

    class FakeRetriever:
        def __init__(self, chunks, index):
            pass

        def search(self, query, top_k=5):
            return passages

    monkeypatch.setattr(answer_cli, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(
        answer_cli,
        "generate_answer",
        lambda question, retrieved: answer_result,
    )


def test_cli_prints_answer_and_citation(tmp_path, monkeypatch, capsys):
    reports, index = setup_inputs(tmp_path)

    passages = [{
        "chunk_id": "chunk-1",
        "filename": "guide.pdf",
        "page_number": 2,
        "text": "Chunking divides documents into smaller passages.",
    }]

    answer = {
        "answer": "Chunking divides documents into smaller passages.",
        "citations": [{
            "filename": "guide.pdf",
            "page_number": 2,
            "chunk_id": "chunk-1",
        }],
    }

    configure_mocks(monkeypatch, passages, answer)

    monkeypatch.setattr(
        "sys.argv",
        [
            "answer_cli",
            "Why is chunking useful?",
            "--reports", str(reports),
            "--index", str(index),
        ],
    )

    assert answer_cli.main() == 0

    output = capsys.readouterr().out
    assert "Chunking divides documents" in output
    assert "guide.pdf, page 2" in output


def test_cli_handles_insufficient_evidence(tmp_path, monkeypatch, capsys):
    reports, index = setup_inputs(tmp_path)

    configure_mocks(
        monkeypatch,
        passages=[],
        answer_result={
            "answer": (
                "I couldn't find enough information in the provided "
                "PDF passages to answer this question reliably."
            ),
            "citations": [],
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "answer_cli",
            "Question without evidence",
            "--reports", str(reports),
            "--index", str(index),
        ],
    )

    assert answer_cli.main() == 0

    output = capsys.readouterr().out
    assert "couldn't find enough information" in output
    assert "No supporting PDF citations available." in output


def test_cli_rejects_missing_index(tmp_path, monkeypatch, capsys):
    reports = tmp_path / "output"
    reports.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "answer_cli",
            "Test question",
            "--reports", str(reports),
            "--index", str(tmp_path / "missing.json"),
        ],
    )

    assert answer_cli.main() == 1
    assert "semantic index not found" in capsys.readouterr().err
