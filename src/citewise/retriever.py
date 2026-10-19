
"""Load processed PDF chunks and retrieve them using BM25."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def get_inflections(word: str) -> set[str]:
    """Return common English singular/plural morphological variants for a word."""
    w = word.lower()
    variants = {w}
    if len(w) <= 2:
        return variants

    if w.endswith("ies") and len(w) > 4:
        variants.add(w[:-3] + "y")
    elif w.endswith("es") and len(w) > 3:
        if w.endswith(("sses", "shes", "ches", "xes", "zes")):
            variants.add(w[:-2])
        else:
            variants.add(w[:-1])
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        variants.add(w[:-1])
    else:
        if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
            variants.add(w[:-1] + "ies")
        elif w.endswith(("s", "sh", "ch", "x", "z")):
            variants.add(w + "es")
        else:
            variants.add(w + "s")
    return variants


def load_chunks(report_paths: Iterable[str | Path]) -> list[dict]:
    """Load chunks and source metadata from CiteWise JSON reports."""
    chunks = []

    for report_path in report_paths:
        path = Path(report_path)

        if not path.is_file():
            raise FileNotFoundError(f"Report does not exist: {path}")

        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON report: {path}") from exc

        if not isinstance(report, dict) or not isinstance(
            report.get("chunks"), list
        ):
            raise ValueError(f"Report has no valid chunks list: {path}")

        document = report.get("document", {})
        filename = document.get("filename", path.name)

        for chunk in report["chunks"]:
            required = (
                "chunk_id",
                "document_id",
                "page_number",
                "text",
            )

            if not isinstance(chunk, dict) or any(
                key not in chunk for key in required
            ):
                raise ValueError(f"Invalid chunk in report: {path}")

            if not isinstance(chunk["text"], str):
                raise ValueError(f"Chunk text must be a string: {path}")

            chunks.append({**chunk, "filename": filename})

    return chunks


class BM25Retriever:
    """Rank PDF chunks using BM25 keyword relevance."""

    def __init__(
        self,
        chunks: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one.")

        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.doc_lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0
        )

        document_frequency = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))

        self.idf = {
            term: math.log(
                1 + (len(self.chunks) - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the highest-scoring chunks for a query."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        raw_terms = tokenize(query)
        if not raw_terms or not self.chunks:
            return []

        query_terms = set(raw_terms)
        seen_variants: set[str] = set()
        term_groups: list[tuple[str, set[str]]] = []
        for term in query_terms:
            if term in seen_variants:
                continue
            variants = get_inflections(term)
            seen_variants.update(variants)
            term_groups.append((term, variants))

        results = []

        for index, chunk in enumerate(self.chunks):
            score = 0.0
            length = self.doc_lengths[index]
            chunk_lower = chunk["text"].lower()

            for primary_term, variants in term_groups:
                present_variants = [
                    v for v in variants if self.term_counts[index].get(v, 0) > 0
                ]
                if not present_variants:
                    continue

                frequency = sum(
                    self.term_counts[index].get(v, 0) for v in present_variants
                )
                inverse_frequency = max(
                    self.idf.get(v, 0.0) for v in present_variants
                )
                if inverse_frequency <= 0:
                    continue

                length_normalization = (
                    1 - self.b
                    + self.b * length / self.average_length
                    if self.average_length
                    else 1
                )

                denominator = (
                    frequency + self.k1 * length_normalization
                )

                term_score = inverse_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )

                # Heading match: chunk has a dedicated section for this term (e.g. "Variables:\n" or "\nVariables:")
                has_heading = any(
                    chunk_lower.startswith(f"{v}:")
                    or f"\n{v}:" in chunk_lower
                    or f"\n{v}\n" in chunk_lower
                    for v in variants
                    if len(v) > 2
                )
                # Definition match: chunk explicitly defines the term
                has_definition = any(
                    f"a {v} is" in chunk_lower
                    or f"an {v} is" in chunk_lower
                    or f"{v} is defined" in chunk_lower
                    or f"{v} are defined" in chunk_lower
                    or f"{v} is named" in chunk_lower
                    for v in variants
                    if len(v) > 2
                )

                if has_heading and has_definition:
                    term_score *= 1.5
                elif has_heading or has_definition:
                    term_score *= 1.25

                score += term_score

            if score > 0:
                results.append({**chunk, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
