"""Persistent query-embedding cache for CiteWise evaluation."""

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

from citewise import embeddings
from citewise.api_limits import wait_for_api_slot
from citewise.embeddings import (
    EMBEDDING_DIMENSIONS,
    MODEL_NAME,
)

DEFAULT_QUERY_CACHE_PATH = "evaluation/data/query_cache.json"
CACHE_VERSION = 1


class QueryCacheMissError(RuntimeError):
    """Raised when a query embedding is required in offline mode but missing from cache."""


class QueryEmbeddingCache:
    """Manages persistent caching of query embeddings for evaluation."""

    def __init__(
        self,
        cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH,
        model: str = MODEL_NAME,
        dimensions: int = EMBEDDING_DIMENSIONS,
        allow_api: bool = False,
        client: Optional[Any] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        self.path = Path(cache_path)
        self.model = model
        self.dimensions = dimensions
        self.allow_api = allow_api
        self.client = client
        self.embed_fn = embed_fn
        self.queries: dict[str, list[float]] = {}

        if self.path.exists():
            self._load()

    def _validate_vector(self, query: str, vector: Any) -> list[float]:
        """Validate that a vector contains finite floats of the expected dimension."""
        if not isinstance(vector, list):
            raise ValueError(f"Cache entry for query '{query}' must be a list.")

        if len(vector) != self.dimensions:
            raise ValueError(
                f"Cache entry for query '{query}' has {len(vector)} dimensions; "
                f"expected {self.dimensions}."
            )

        validated: list[float] = []
        for val in vector:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"Cache entry for query '{query}' contains non-numeric value: {val!r}."
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Cache entry for query '{query}' contains non-finite value: {val!r}."
                )
            validated.append(float(val))

        return validated

    def _load(self) -> None:
        """Load and validate existing cache file."""
        try:
            content = self.path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid query cache JSON at {self.path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Query cache at {self.path} must be a JSON object.")

        cached_model = data.get("model")
        if cached_model != self.model:
            raise ValueError(
                f"Query cache model mismatch: expected '{self.model}', got '{cached_model}'."
            )

        cached_dims = data.get("dimensions")
        if cached_dims != self.dimensions:
            raise ValueError(
                f"Query cache dimensions mismatch: expected {self.dimensions}, got {cached_dims}."
            )

        raw_queries = data.get("queries")
        if not isinstance(raw_queries, dict):
            raise ValueError(f"Query cache 'queries' field must be an object at {self.path}.")

        loaded: dict[str, list[float]] = {}
        for q, entry in raw_queries.items():
            if not isinstance(q, str):
                raise ValueError(f"Query cache key must be a string: {q!r}.")

            if isinstance(entry, dict):
                # Entry with metadata
                entry_model = entry.get("model", self.model)
                entry_dims = entry.get("dimensions", self.dimensions)
                if entry_model != self.model:
                    raise ValueError(
                        f"Query cache entry '{q}' model mismatch: {entry_model} != {self.model}."
                    )
                if entry_dims != self.dimensions:
                    raise ValueError(
                        f"Query cache entry '{q}' dimensions mismatch: {entry_dims} != {self.dimensions}."
                    )
                vector = entry.get("embedding")
            elif isinstance(entry, list):
                vector = entry
            else:
                raise ValueError(f"Invalid cache entry format for query '{q}'.")

            loaded[q] = self._validate_vector(q, vector)

        self.queries = loaded

    def save(self) -> None:
        """Atomically persist cache entries to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(self.path.name + ".tmp")

        payload: dict[str, Any] = {
            "version": CACHE_VERSION,
            "model": self.model,
            "dimensions": self.dimensions,
            "queries": {
                q: {
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "embedding": vec,
                }
                for q, vec in self.queries.items()
            },
        }

        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def get_embedding(self, query: str) -> list[float]:
        """Retrieve embedding for a query, or compute and cache if API opt-in is enabled."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if query in self.queries:
            return self.queries[query]

        if not self.allow_api:
            raise QueryCacheMissError(
                f"Query embedding cache miss for query: {query!r}.\n"
                f"In offline mode, live API calls are forbidden.\n"
                f"To populate missing query embeddings, run with --allow-query-embedding-api."
            )

        # Opted-in API path: pace request and call embedder
        wait_for_api_slot()
        if self.embed_fn is not None:
            raw_vector = self.embed_fn(query)
        else:
            raw_vector = embeddings.embed_query(query, client=self.client)

        vector = self._validate_vector(query, raw_vector)
        self.queries[query] = vector
        self.save()  # Immediately save so interrupted runs do not lose work
        return vector

    def set_embedding(self, query: str, vector: list[float]) -> None:
        """Explicitly set and persist an embedding for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        validated = self._validate_vector(query, vector)
        if query in self.queries and self.queries[query] == validated:
            return

        self.queries[query] = validated
        self.save()


def populate_cache_from_dataset(
    dataset_path: str | Path,
    cache_path: str | Path = DEFAULT_QUERY_CACHE_PATH,
    client: Optional[Any] = None,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> int:
    """Populate query embeddings for all cases in an evaluation dataset."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases", [])
    else:
        raise ValueError("Invalid dataset: must be a list or object.")

    cache = QueryEmbeddingCache(
        cache_path=cache_path,
        allow_api=True,
        client=client,
        embed_fn=embed_fn,
    )

    populated_count = 0
    for case in cases:
        question = case.get("question")
        if not question or not isinstance(question, str):
            continue
        if question not in cache.queries:
            cache.get_embedding(question)
            populated_count += 1

    return populated_count


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Populate or inspect CiteWise evaluation query embedding cache."
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/dataset.json",
        help="Path to evaluation dataset JSON.",
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_QUERY_CACHE_PATH,
        help=f"Path to query embedding cache JSON (default: {DEFAULT_QUERY_CACHE_PATH}).",
    )
    parser.add_argument(
        "--allow-query-embedding-api",
        action="store_true",
        help="Opt-in to calling Gemini API to embed missing queries.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"Error: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    cache = QueryEmbeddingCache(
        cache_path=args.cache,
        allow_api=args.allow_query_embedding_api,
    )

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases", [])
    else:
        cases = []

    questions = [c["question"] for c in cases if isinstance(c, dict) and "question" in c]

    missing = [q for q in questions if q not in cache.queries]

    print(f"Cache location: {cache.path}")
    print(f"Model:          {cache.model}")
    print(f"Dimensions:     {cache.dimensions}")
    print(f"Cached queries: {len(cache.queries)}")
    print(f"Total dataset:  {len(questions)}")
    print(f"Missing:        {len(missing)}")

    if not missing:
        print("All dataset questions are cached.")
        return 0

    if not args.allow_query_embedding_api:
        print(
            f"Error: {len(missing)} query embeddings are missing from the cache. "
            "Pass --allow-query-embedding-api to populate them.",
            file=sys.stderr,
        )
        return 1

    print(f"Populating {len(missing)} missing query embeddings...")
    count = populate_cache_from_dataset(dataset_path, cache_path=args.cache)
    print(f"Successfully populated {count} query embeddings.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
