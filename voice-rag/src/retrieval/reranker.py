"""Lightweight reranker placeholder."""

from __future__ import annotations


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    query_terms = set(query.lower().split())
    rescored: list[dict] = []

    for hit in hits:
        overlap = len(query_terms.intersection(set(hit["text"].lower().split())))
        bonus = overlap * 0.01
        rescored.append({**hit, "score": hit.get("score", 0.0) + bonus})

    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]
