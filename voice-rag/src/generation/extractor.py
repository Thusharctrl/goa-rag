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


_STOPWORDS = {
    # English
    "what", "how", "why", "who", "when", "where", "which", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "with", "this", "that",
    "do", "does", "did", "can", "could", "would", "should", "it", "they", "them",
    "what's", "how's", "who's", "there", "their", "here", "be", "been", "being",
    # Hindi
    "क्या", "है", "हैं", "था", "थी", "थे", "कैसे", "कौन", "क्यों", "कब", "कहाँ",
    "में", "की", "के", "का", "और", "से", "को", "पर", "यह", "वह", "कि", "लिए", "एक",
    "तो", "भी", "ही", "जो", "कर", "ने"
}

def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split, and remove stopwords."""
    cleaned = text.lower()
    for p in ".,!?;:\"'()[]{}|-":
        cleaned = cleaned.replace(p, " ")
    
    terms = set()
    for t in cleaned.split():
        if t not in _STOPWORDS and len(t) > 1:
            terms.add(t)
    return terms


def _score(query_terms: set[str], sentence: str) -> float:
    s_terms = _tokenize(sentence)
    if not query_terms or not s_terms:
        return 0.0
    
    intersection = query_terms & s_terms
    if not intersection:
        return 0.0
        
    # Emphasize recall (finding the query terms) rather than strict Jaccard
    # which heavily penalizes informative longer sentences.
    recall = len(intersection) / len(query_terms)
    density = len(intersection) / (len(s_terms) + 5.0)
    
    return (recall * 0.8) + (density * 0.2)


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

    query_terms = _tokenize(query)

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
