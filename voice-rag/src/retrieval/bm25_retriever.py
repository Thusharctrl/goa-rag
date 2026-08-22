"""BM25 lexical retrieval over the same corpus loaded by the HNSW index.

Uses rank_bm25 (already installed) and piggybacks on the HNSW corpus
so we don't re-read Chroma twice.
"""
from __future__ import annotations

from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.retrieval.hnsw_index import _build_hnsw


@lru_cache(maxsize=1)
def _build_bm25_by_language() -> dict[str, tuple[BM25Okapi, list[int]]]:
    """Return {language -> (BM25Okapi, [original_indices])}."""
    corpus = _build_hnsw()  # reuses cached build
    lang_docs: dict[str, list[tuple[int, str]]] = {}
    for i, (text, meta) in enumerate(zip(corpus.texts, corpus.metadatas)):
        lang = meta.get("language", "en")
        lang_docs.setdefault(lang, []).append((i, text))

    result: dict[str, tuple[BM25Okapi, list[int]]] = {}
    for lang, pairs in lang_docs.items():
        indices = [p[0] for p in pairs]
        tokenized = [p[1].lower().split() for p in pairs]
        result[lang] = (BM25Okapi(tokenized), indices)

    return result


def bm25_query(
    query: str,
    *,
    language: str,
    top_k: int = 20,
) -> list[dict]:
    """Return BM25-scored hits for query in the given language."""
    corpus = _build_hnsw()
    by_lang = _build_bm25_by_language()

    if language not in by_lang:
        return []

    bm25, indices = by_lang[language]
    scores = bm25.get_scores(query.lower().split())

    # Pair with original corpus index
    ranked = sorted(
        zip(indices, scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    hits: list[dict] = []
    for orig_idx, score in ranked:
        if score <= 0:
            continue
        hits.append(
            {
                "text": corpus.texts[orig_idx],
                "score": float(score),
                "metadata": corpus.metadatas[orig_idx],
            }
        )
    return hits


def warmup_bm25() -> None:
    """Force BM25 index build at startup."""
    _build_bm25_by_language()
