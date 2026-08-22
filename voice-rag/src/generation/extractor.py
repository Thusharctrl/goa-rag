"""Extractive answer selection.

Picks the most query-relevant sentence from the retrieved passages without
any LLM call. This is the fast-path answer: grounded, cited, <10 ms.
"""
from __future__ import annotations

import re


_SENT_RE = re.compile(r"(?<=[.!?।])\s+")


def _sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _score(query_terms: set[str], sentence: str) -> float:
    s_terms = set(sentence.lower().split())
    if not s_terms:
        return 0.0
    return len(query_terms & s_terms) / len(s_terms | query_terms)


def extract_answer(
    query: str,
    hits: list[dict],
    *,
    max_sentences: int = 3,
) -> tuple[str, float]:
    """Return (answer_text, confidence_score).

    Selects the best-scoring sentences from the top retrieved passages.
    Falls back to the leading text of the top hit if no sentence scores above 0.
    """
    if not hits:
        return "I do not have enough information to answer that question.", 0.0

    query_terms = {
        t.lower().strip(".,!?;:\"'()[]{}")
        for t in query.split()
        if len(t) >= 3
    }

    best: list[tuple[float, str]] = []

    for hit in hits[:5]:  # only scan top-5 passages
        passage_score = hit.get("score", 0.0)
        for sent in _sentences(hit["text"]):
            s = _score(query_terms, sent)
            weighted = s * 0.7 + passage_score * 0.3
            best.append((weighted, sent))

    if not best:
        top = hits[0]["text"]
        return top[:500], hits[0].get("score", 0.0)

    best.sort(key=lambda x: x[0], reverse=True)
    top_score, top_sent = best[0]

    # Collect up to max_sentences that are still highly relevant
    selected = [top_sent]
    threshold = top_score * 0.6
    for score, sent in best[1:]:
        if score >= threshold and sent not in selected:
            selected.append(sent)
        if len(selected) >= max_sentences:
            break

    answer = " ".join(selected)
    return answer, float(top_score)
