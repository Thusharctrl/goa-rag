"""In-memory HNSW index built from the existing Chroma collection.

Loaded once at startup; subsequent queries are pure in-process RAM lookups,
eliminating the IPC overhead of Chroma's gRPC/SQLite path.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import hnswlib

from src.config import settings


class _HNSWCorpus(NamedTuple):
    index: hnswlib.Index
    texts: list[str]
    metadatas: list[dict]
    dim: int


@lru_cache(maxsize=1)
def _build_hnsw() -> _HNSWCorpus:
    """Load all vectors from Chroma and build an in-memory HNSW graph."""
    import chromadb

    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    col = client.get_collection(name=settings.collection_name)
    total = col.count()

    if total == 0:
        raise RuntimeError(
            "Chroma collection is empty. Run scripts/build_index.py first."
        )

    # Pull everything in one batch (11k items is fine in RAM)
    batch_size = 5000
    all_embeddings: list[list[float]] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []

    for offset in range(0, total, batch_size):
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "documents", "metadatas"],
        )
        embs = result["embeddings"]
        docs = result["documents"]
        metas = result["metadatas"]
        # Chroma may return numpy arrays; extend with explicit None checks
        if embs is not None:
            all_embeddings.extend(embs.tolist() if hasattr(embs, "tolist") else list(embs))
        if docs is not None:
            all_texts.extend(list(docs))
        if metas is not None:
            all_metas.extend(list(metas))

    dim = len(all_embeddings[0])
    n = len(all_embeddings)

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.set_ef(50)
    idx.add_items(np.array(all_embeddings, dtype=np.float32), list(range(n)))

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[hnsw] Loaded {n} vectors (dim={dim}) in {elapsed:.0f} ms")
    return _HNSWCorpus(index=idx, texts=all_texts, metadatas=all_metas, dim=dim)


def warmup_hnsw() -> None:
    """Pre-build the index so first query is not penalised."""
    _build_hnsw()


def hnsw_query(
    query_vec: list[float],
    *,
    language: str,
    top_k: int = 20,
) -> list[dict]:
    """Return up to top_k candidates filtered to `language`, sorted by cosine similarity."""
    corpus = _build_hnsw()
    # Over-fetch to have enough after language filter
    k = min(top_k * 6, corpus.index.get_current_count())
    labels, distances = corpus.index.knn_query(
        np.array([query_vec], dtype=np.float32), k=k
    )

    hits: list[dict] = []
    for label, dist in zip(labels[0], distances[0]):
        meta = corpus.metadatas[label]
        if meta.get("language") != language:
            continue
        hits.append(
            {
                "text": corpus.texts[label],
                "score": float(1.0 - dist),  # cosine distance → similarity
                "metadata": meta,
            }
        )
        if len(hits) >= top_k:
            break

    return hits
