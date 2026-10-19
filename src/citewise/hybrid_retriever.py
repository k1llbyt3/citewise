
"""Combine BM25 and semantic search rankings using RRF."""

from typing import Any, Optional

from citewise.retriever import BM25Retriever
from citewise.semantic_retriever import SemanticRetriever


DEFAULT_RRF_K = 60


class HybridRetriever:
    """Combine keyword and semantic rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        index: dict[str, Any],
        client=None,
        rrf_k: int = DEFAULT_RRF_K,
        query_cache: Optional[Any] = None,
    ):
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")

        self.bm25 = BM25Retriever(chunks)
        if query_cache is not None:
            try:
                self.semantic = SemanticRetriever(
                    index, client=client, query_cache=query_cache
                )
            except TypeError:
                self.semantic = SemanticRetriever(index, client=client)
        else:
            self.semantic = SemanticRetriever(index, client=client)
        self.rrf_k = rrf_k
        self.query_cache = query_cache

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return chunks ranked by the combined BM25 and semantic rankings."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if not query.strip():
            return []

        # Get the full rankings so RRF can combine both lists.
        keyword_results = (
            self.bm25.search(query, top_k=len(self.bm25.chunks))
            if self.bm25.chunks
            else []
        )
        semantic_results = (
            self.semantic.search(query, top_k=len(self.semantic.items))
            if self.semantic.items
            else []
        )

        combined: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result["chunk_id"]
            combined[chunk_id] = {
                **result,
                "bm25_rank": rank,
                "bm25_score": result.get("score"),
                "semantic_rank": None,
                "semantic_score": None,
                "hybrid_score": 1.0 / (self.rrf_k + rank),
            }

        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "semantic_rank": rank,
                    "semantic_score": result.get("score"),
                    "hybrid_score": 0.0,
                }

            combined[chunk_id]["semantic_rank"] = rank
            combined[chunk_id]["semantic_score"] = result.get("score")
            combined[chunk_id]["hybrid_score"] += 1.0 / (
                self.rrf_k + rank
            )

        results = list(combined.values())
        results.sort(
            key=lambda result: (
                result["hybrid_score"],
                -(result.get("bm25_rank") or float("inf")),
            ),
            reverse=True,
        )

        return results[:top_k]
