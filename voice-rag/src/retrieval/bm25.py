"""Optional lexical retrieval helper (not used by the current MVP pipeline)."""

from __future__ import annotations


class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - optional extension
            raise ImportError(
                "BM25 retrieval is optional. Install rank-bm25 to use this helper."
            ) from exc

        tokenized = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {"text": self._documents[idx], "score": float(score), "metadata": {}}
            for idx, score in ranked
        ]
