
"""Build and persist a local semantic embedding index."""

import json
import math
from pathlib import Path
from typing import Any

from citewise.embeddings import (
    DOCUMENT_TASK,
    EMBEDDING_DIMENSIONS,
    MODEL_NAME,
    embed_texts,
)


INDEX_VERSION = 1
REQUIRED_FIELDS = ("chunk_id", "document_id", "page_number", "text")


def build_embedding_index(
    chunks: list[dict[str, Any]],
    client=None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Generate vectors and preserve the metadata for each chunk."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list.")

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Every chunk must be a dictionary.")

        if any(field not in chunk for field in REQUIRED_FIELDS):
            raise ValueError("A chunk is missing required metadata.")

        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError("Chunk text must not be empty.")

    texts = [chunk["text"] for chunk in chunks]

    vectors = embed_texts(
        texts,
        task_type=DOCUMENT_TASK,
        batch_size=batch_size,
        client=client,
    )

    items = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        items.append({
            **chunk,
            "embedding": vector,
        })

    return {
        "index_version": INDEX_VERSION,
        "model": MODEL_NAME,
        "dimensions": EMBEDDING_DIMENSIONS,
        "items": items,
    }


def save_embedding_index(
    index: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the index as JSON, replacing the destination atomically."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        temporary_path.write_text(
            json.dumps(index, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_embedding_index(
    input_path: str | Path,
) -> dict[str, Any]:
    """Load an index and validate its basic structure and vectors."""
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(f"Embedding index not found: {path}")

    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid embedding index JSON: {path}") from exc

    if not isinstance(index, dict):
        raise ValueError("Embedding index must be a JSON object.")

    if index.get("index_version") != INDEX_VERSION:
        raise ValueError("Unsupported embedding index version.")

    if index.get("model") != MODEL_NAME:
        raise ValueError("Embedding index uses an unexpected model.")

    if index.get("dimensions") != EMBEDDING_DIMENSIONS:
        raise ValueError("Embedding index has unexpected dimensions.")

    items = index.get("items")

    if not isinstance(items, list):
        raise ValueError("Embedding index must contain an items list.")

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every index item must be an object.")

        if any(field not in item for field in REQUIRED_FIELDS):
            raise ValueError("An index item is missing required metadata.")

        if not isinstance(item["text"], str):
            raise ValueError("Indexed chunk text must be a string.")

        vector = item.get("embedding")

        if not isinstance(vector, list):
            raise ValueError("An index item has no valid embedding vector.")

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError("An index item has the wrong vector dimensions.")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("An embedding contains an invalid numeric value.")

    return index
