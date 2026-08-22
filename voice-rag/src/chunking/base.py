from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    language: str
    query_id: int
    row_index: int
    passage_index: int
    parent_id: str | None = None
