from src.embeddings.index import VectorIndex


class VectorRetriever:
    def __init__(self, index: VectorIndex | None = None) -> None:
        self._index = index or VectorIndex()

    def retrieve(self, query: str, *, language: str, top_k: int) -> list[dict]:
        hits = self._index.query(query, language=language, top_k=top_k)
        seen: set[str] = set()
        deduped: list[dict] = []

        for hit in hits:
            parent_id = hit["metadata"].get("parent_id") or hit["metadata"].get("chunk_id", hit["text"])
            key = str(parent_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)

        return deduped
