from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from src.chunking.base import Chunk
from src.config import settings
from src.embeddings.model import embed_texts


class VectorIndex:
    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def reset(self) -> None:
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "strategy": chunk.strategy,
                    "language": chunk.language,
                    "query_id": chunk.query_id,
                    "row_index": chunk.row_index,
                    "passage_index": chunk.passage_index,
                    "parent_id": chunk.parent_id or "",
                }
                for chunk in chunks
            ],
        )

    def query(
        self,
        query_text: str,
        *,
        language: str,
        top_k: int,
    ) -> list[dict]:
        query_embedding = embed_texts([query_text])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"language": language},
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
            score = 1.0 - float(distance)
            hits.append(
                {
                    "text": doc,
                    "score": score,
                    "metadata": meta,
                }
            )
        return hits
