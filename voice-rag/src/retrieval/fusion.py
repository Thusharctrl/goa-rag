"""Score fusion helper for hybrid retrieval."""

from __future__ import annotations


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads[key] = item

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        {**payloads[text], "score": score}
        for text, score in ranked
    ]
