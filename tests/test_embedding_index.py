
import json
from types import SimpleNamespace

import pytest

from citewise.embedding_index import (
    INDEX_VERSION,
    build_embedding_index,
    load_embedding_index,
    save_embedding_index,
)
from citewise.embeddings import EMBEDDING_DIMENSIONS, MODEL_NAME


class FakeEmbeddingClient:
    def __init__(self):
        self.models = self
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[float(len(text))] * EMBEDDING_DIMENSIONS
                )
                for text in kwargs["contents"]
            ]
        )


def sample_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "page_number": 1,
            "text": "Python is used for data analysis.",
            "filename": "guide.pdf",
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "page_number": 2,
            "text": "Data can be stored in databases.",
            "filename": "guide.pdf",
        },
    ]


def test_build_index_preserves_metadata_and_vectors():
    client = FakeEmbeddingClient()

    index = build_embedding_index(sample_chunks(), client=client)

    assert index["index_version"] == INDEX_VERSION
    assert index["model"] == MODEL_NAME
    assert index["dimensions"] == EMBEDDING_DIMENSIONS
    assert len(index["items"]) == 2
    assert index["items"][0]["filename"] == "guide.pdf"
    assert index["items"][0]["page_number"] == 1
    assert len(index["items"][0]["embedding"]) == EMBEDDING_DIMENSIONS
    assert len(client.calls) == 1


def test_build_index_returns_empty_index_for_no_chunks():
    client = FakeEmbeddingClient()

    index = build_embedding_index([], client=client)

    assert index["items"] == []
    assert client.calls == []


def test_build_index_rejects_non_list():
    with pytest.raises(TypeError, match="chunks must be a list"):
        build_embedding_index("not a list")


def test_build_index_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing required metadata"):
        build_embedding_index([{"text": "Some passage"}])


def test_build_index_rejects_blank_text():
    chunk = sample_chunks()[0]
    chunk["text"] = "   "

    with pytest.raises(ValueError, match="must not be empty"):
        build_embedding_index([chunk])


def test_save_and_load_index_round_trip(tmp_path):
    client = FakeEmbeddingClient()
    index = build_embedding_index(sample_chunks(), client=client)
    path = tmp_path / "nested" / "semantic_index.json"

    save_embedding_index(index, path)
    loaded = load_embedding_index(path)

    assert loaded == index
    assert path.is_file()


def test_load_index_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_embedding_index(tmp_path / "missing.json")


def test_load_index_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid embedding index JSON"):
        load_embedding_index(path)


@pytest.mark.parametrize("payload", [[], None, "invalid"])
def test_load_index_rejects_non_object_json(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_embedding_index(path)


def test_load_index_rejects_wrong_version(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({
            "index_version": 999,
            "model": MODEL_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "items": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version"):
        load_embedding_index(path)


def test_load_index_rejects_wrong_vector_dimensions(tmp_path):
    client = FakeEmbeddingClient()
    index = build_embedding_index(sample_chunks(), client=client)
    index["items"][0]["embedding"] = [0.1, 0.2]

    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong vector dimensions"):
        load_embedding_index(path)


def test_load_index_rejects_non_finite_vector_values(tmp_path):
    client = FakeEmbeddingClient()
    index = build_embedding_index(sample_chunks(), client=client)
    index["items"][0]["embedding"][0] = float("nan")

    path = tmp_path / "index.json"
    path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid numeric value"):
        load_embedding_index(path)
