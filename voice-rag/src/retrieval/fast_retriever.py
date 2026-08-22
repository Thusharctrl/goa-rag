"""Fast retriever: HNSW dense + BM25 lexical fused with Reciprocal Rank Fusion.

Entry point for the hot path. All data structures are in-memory; this module
has no Chroma calls after startup.
"""
from __future__ import annotations

from src.embeddings.onnx_model import onnx_embed
from src.retrieval.bm25_retriever import bm25_query
from src.retrieval.hnsw_index import hnsw_query


def _rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = item["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads[key] = item

    ranked = sorted(scores.items(), key=lambda p: p[1], reverse=True)
    return [{**payloads[text], "score": rrf_score} for text, rrf_score in ranked]


def fast_retrieve(
    query: str,
    *,
    language: str,
    top_k: int,
) -> list[dict]:
    """Embed + HNSW + BM25 + RRF. Returns up to top_k deduplicated hits."""
    query_vec = onnx_embed([query])[0]

    dense_hits = hnsw_query(query_vec, language=language, top_k=top_k * 3)
    lexical_hits = bm25_query(query, language=language, top_k=top_k * 3)

    fused = _rrf([dense_hits, lexical_hits])

    # Deduplicate by parent_id / chunk_id
    seen: set[str] = set()
    deduped: list[dict] = []
    for hit in fused:
        meta = hit.get("metadata", {})
        key = str(meta.get("parent_id") or meta.get("chunk_id") or hit["text"][:80])
        if key not in seen:
            seen.add(key)
            deduped.append(hit)
        if len(deduped) >= top_k:
            break

    return deduped


def warmup_fast_retriever() -> None:
    """Call from startup to pre-build all in-memory structures."""
    from src.retrieval.hnsw_index import warmup_hnsw
    from src.retrieval.bm25_retriever import warmup_bm25
    from src.embeddings.onnx_model import warmup_onnx

    warmup_onnx()
    warmup_hnsw()
    warmup_bm25()
