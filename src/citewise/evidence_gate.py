"""Local pre-generation evidence gate for CiteWise.

Evaluates retrieval results locally before invoking LLM generation to reject
clearly unsupported queries, avoid redundant API failures, and conserve quota.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Optional

# Standard English stopwords to filter function words from content overlap calculation
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "shouldn't", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves",
}


def extract_content_tokens(text: str) -> set[str]:
    """Extract lowercased alphanumeric content tokens excluding standard stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


@dataclass
class EvidenceGateResult:
    """Structured decision returned by the local evidence gate."""

    decision: Literal["pass", "reject", "borderline"]
    reason: str
    explanation: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_generate(self) -> bool:
        """Return True if generation should proceed (pass or borderline)."""
        return self.decision in ("pass", "borderline")


def evaluate_evidence_gate(
    query: str,
    passages: list[dict[str, Any]],
    retriever_type: str = "bm25",
) -> EvidenceGateResult:
    """Evaluate retrieval signals locally to decide whether to invoke generation.

    Distinguishes three outcomes:
      - pass: Evidence is sufficiently relevant to proceed with generation.
      - reject: Evidence is clearly weak or absent; generation should be skipped.
      - borderline: Signals are ambiguous (e.g. potential paraphrase); generation
                    proceeds to avoid overconfident rejection, with caution logged.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return EvidenceGateResult(
            decision="reject",
            reason="Empty query",
            explanation="The query contains no searchable terms.",
            metrics={"query_terms_count": 0, "passages_count": len(passages)},
        )

    if not passages:
        return EvidenceGateResult(
            decision="reject",
            reason="Empty retrieval",
            explanation="No matching passages were found in the selected documents.",
            metrics={"query_terms_count": len(extract_content_tokens(cleaned_query)), "passages_count": 0},
        )

    content_tokens = extract_content_tokens(cleaned_query)
    all_passage_text = " ".join(p.get("text", "") for p in passages)
    all_passage_tokens = extract_content_tokens(all_passage_text)

    top_passage = passages[0]
    top_passage_text = top_passage.get("text", "")
    top_passage_tokens = extract_content_tokens(top_passage_text)

    if content_tokens:
        matched_tokens = content_tokens.intersection(all_passage_tokens)
        overlap_ratio = len(matched_tokens) / len(content_tokens)
        top_matched_tokens = content_tokens.intersection(top_passage_tokens)
        top_overlap_ratio = len(top_matched_tokens) / len(content_tokens)
    else:
        # Query consists purely of stopwords or numbers
        matched_tokens = set()
        overlap_ratio = 0.0
        top_matched_tokens = set()
        top_overlap_ratio = 0.0

    retriever_mode = retriever_type.lower()

    # Extract score signals
    top_score = top_passage.get("score")
    top_bm25_score = top_passage.get("bm25_score")
    top_semantic_score = top_passage.get("semantic_score")

    # In single-mode semantic retriever, score is cosine similarity
    if retriever_mode == "semantic" and top_score is not None:
        top_semantic_score = top_score
    elif retriever_mode == "bm25" and top_score is not None:
        top_bm25_score = top_score

    score_separation = None
    if len(passages) > 1 and "score" in top_passage and "score" in passages[1]:
        s0 = top_passage.get("score")
        s1 = passages[1].get("score")
        if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
            score_separation = round(float(s0 - s1), 4)

    combined_evidence = bool(
        len(passages) > 1
        and overlap_ratio > top_overlap_ratio
        and len(matched_tokens) > len(top_matched_tokens)
    )

    metrics: dict[str, Any] = {
        "retriever_type": retriever_mode,
        "query_terms_count": len(content_tokens),
        "matched_terms_count": len(matched_tokens),
        "lexical_overlap_ratio": round(overlap_ratio, 4),
        "top_passage_overlap_ratio": round(top_overlap_ratio, 4),
        "combined_evidence": combined_evidence,
        "top_score": top_score,
        "bm25_score": top_bm25_score,
        "semantic_similarity": top_semantic_score,
        "score_separation": score_separation,
        "passages_count": len(passages),
    }

    # If query has no content tokens at all (e.g. "What is it?"):
    if not content_tokens:
        return EvidenceGateResult(
            decision="borderline",
            reason="Query lacks distinctive content words",
            explanation="The query consists of common words only; evidence cannot be verified lexically.",
            metrics=metrics,
        )

    # --- Mode: BM25 ---
    if retriever_mode == "bm25":
        bm_val = top_bm25_score if top_bm25_score is not None else (top_score if isinstance(top_score, (int, float)) else 0.0)

        if overlap_ratio == 0.0 or bm_val <= 0.0:
            return EvidenceGateResult(
                decision="reject",
                reason="Zero lexical overlap",
                explanation=(
                    f"None of the query's key terms ({', '.join(sorted(content_tokens)[:6])}) "
                    f"appear in the retrieved passages. BM25 score ({bm_val:.2f}) was derived solely from function words."
                ),
                metrics=metrics,
            )

        if bm_val >= 12.0 and overlap_ratio >= 0.30:
            return EvidenceGateResult(
                decision="pass",
                reason="Strong lexical support",
                explanation=f"Top BM25 score of {bm_val:.2f} with {int(overlap_ratio * 100)}% query term overlap.",
                metrics=metrics,
            )

        if combined_evidence and overlap_ratio >= 0.40:
            return EvidenceGateResult(
                decision="pass",
                reason="Combined multi-passage evidence",
                explanation=f"Evidence is jointly supported across {len(passages)} retrieved passages with {int(overlap_ratio * 100)}% query term coverage.",
                metrics=metrics,
            )

        if overlap_ratio >= 0.50:
            return EvidenceGateResult(
                decision="pass",
                reason="Substantial query term overlap",
                explanation=f"{int(overlap_ratio * 100)}% of query terms matched retrieved text.",
                metrics=metrics,
            )

        if bm_val < 5.0 and overlap_ratio < 0.20:
            return EvidenceGateResult(
                decision="reject",
                reason="Insufficient lexical relevance",
                explanation=f"BM25 score ({bm_val:.2f}) and term overlap ({int(overlap_ratio * 100)}%) are too low to provide reliable evidence.",
                metrics=metrics,
            )

        return EvidenceGateResult(
            decision="borderline",
            reason="Borderline lexical evidence",
            explanation=f"Partial term overlap ({int(overlap_ratio * 100)}%) with modest BM25 score ({bm_val:.2f}). Forwarding to generator.",
            metrics=metrics,
        )

    # --- Mode: Semantic or Hybrid ---
    sem_sim = top_semantic_score if top_semantic_score is not None else (
        top_score if retriever_mode == "semantic" and isinstance(top_score, (int, float)) and top_score <= 1.0 else None
    )

    # High semantic confidence
    if sem_sim is not None and sem_sim >= 0.70:
        return EvidenceGateResult(
            decision="pass",
            reason="High semantic confidence",
            explanation=f"Semantic similarity of {sem_sim:.3f} indicates strong conceptual alignment.",
            metrics=metrics,
        )

    # Substantial lexical overlap or combined multi-passage evidence
    if combined_evidence and overlap_ratio >= 0.40:
        return EvidenceGateResult(
            decision="pass",
            reason="Combined multi-passage evidence",
            explanation=f"Evidence is jointly supported across {len(passages)} retrieved passages with {int(overlap_ratio * 100)}% query term coverage.",
            metrics=metrics,
        )

    if overlap_ratio >= 0.40:
        return EvidenceGateResult(
            decision="pass",
            reason="Substantial query term overlap",
            explanation=f"{int(overlap_ratio * 100)}% of query terms matched retrieved passages.",
            metrics=metrics,
        )

    # Zero lexical overlap case
    if overlap_ratio == 0.0:
        if sem_sim is not None and sem_sim >= 0.65:
            # Possible paraphrase with distinct vocabulary
            return EvidenceGateResult(
                decision="borderline",
                reason="Potential semantic paraphrase",
                explanation=(
                    f"Zero lexical overlap, but semantic similarity of {sem_sim:.3f} suggests "
                    "a possible paraphrase. Forwarding to generator."
                ),
                metrics=metrics,
            )
        # Clear out-of-context query
        sim_desc = f"{sem_sim:.3f}" if sem_sim is not None else "N/A"
        return EvidenceGateResult(
            decision="reject",
            reason="Uncorrelated retrieval",
            explanation=(
                f"No query terms matched retrieved passages and semantic similarity ({sim_desc}) "
                "is insufficient to attempt grounded generation."
            ),
            metrics=metrics,
        )

    # Low semantic similarity with very low overlap
    if sem_sim is not None and sem_sim < 0.35 and overlap_ratio < 0.20:
        return EvidenceGateResult(
            decision="reject",
            reason="Weak evidence signals",
            explanation=(
                f"Both semantic similarity ({sem_sim:.3f}) and lexical overlap ({int(overlap_ratio * 100)}%) "
                "are below reliable evidence thresholds."
            ),
            metrics=metrics,
        )

    # Borderline default for semantic/hybrid
    sim_desc = f"{sem_sim:.3f}" if sem_sim is not None else "N/A"
    return EvidenceGateResult(
        decision="borderline",
        reason="Borderline relevance",
        explanation=(
            f"Evidence signals are moderate (overlap: {int(overlap_ratio * 100)}%, similarity: {sim_desc}). "
            "Forwarding to generator to assess context."
        ),
        metrics=metrics,
    )
