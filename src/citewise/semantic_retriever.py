
"""Retrieve PDF passages using semantic similarity."""

import math
from typing import Any, Optional

from citewise.embeddings import EMBEDDING_DIMENSIONS, embed_query


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Measure how similar two vectors are by their direction."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    if not vector_a:
        raise ValueError("Vectors must not be empty.")

    for vector in (vector_a, vector_b):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Vectors must contain finite numeric values.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(value * value for value in vector_a))
    magnitude_b = math.sqrt(sum(value * value for value in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class SemanticRetriever:
    """Rank indexed PDF chunks by similarity to a search query."""

    def __init__(
        self,
        index: dict[str, Any],
        client=None,
        query_cache: Optional[Any] = None,
    ):
        if not isinstance(index, dict):
            raise TypeError("index must be a dictionary.")

        if index.get("dimensions") != EMBEDDING_DIMENSIONS:
            raise ValueError("Index has unexpected embedding dimensions.")

        items = index.get("items")

        if not isinstance(items, list):
            raise ValueError("Index must contain an items list.")

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Every index item must be a dictionary.")

            required = ("chunk_id", "document_id", "page_number", "text")
            if any(field not in item for field in required):
                raise ValueError("An index item is missing required metadata.")

            if not isinstance(item["text"], str):
                raise ValueError("Indexed chunk text must be a string.")

            vector = item.get("embedding")

            if not isinstance(vector, list):
                raise ValueError("An index item has no valid embedding.")

            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError("An index item has the wrong vector dimensions.")

            # Validate numeric values now, rather than during every search.
            cosine_similarity(vector, vector)

        self.items = items
        self.client = client
        self.query_cache = query_cache

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top matching chunks, including source metadata."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        if not query.strip() or not self.items:
            return []

        if self.query_cache is not None:
            query_vector = self.query_cache.get_embedding(query)
        else:
            query_vector = embed_query(query, client=self.client)

        if len(query_vector) != EMBEDDING_DIMENSIONS:
            raise ValueError("Query embedding has the wrong dimensions.")

        results = []

        for item in self.items:
            score = cosine_similarity(query_vector, item["embedding"])
            results.append({**item, "score": score})

        results.sort(key=lambda result: result["score"], reverse=True)

        return results[:top_k]
