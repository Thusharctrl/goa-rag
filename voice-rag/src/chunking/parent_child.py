from src.chunking.base import Chunk
from src.chunking.sentence import chunk_sentences


def chunk_parent_child(
    text: str,
    *,
    language: str,
    query_id: int,
    passage_index: int,
) -> list[Chunk]:
    if not text:
        return []

    parent_id = f"{language}:{query_id}:{passage_index}:parent"
    parent = Chunk(
        chunk_id=parent_id,
        text=text,
        strategy="parent_child",
        language=language,
        query_id=query_id,
        passage_index=passage_index,
        parent_id=parent_id,
    )

    children = []
    for child in chunk_sentences(
        text,
        strategy="parent_child",
        language=language,
        query_id=query_id,
        passage_index=passage_index,
    ):
        children.append(
            Chunk(
                chunk_id=child.chunk_id,
                text=child.text,
                strategy=child.strategy,
                language=child.language,
                query_id=child.query_id,
                passage_index=child.passage_index,
                parent_id=parent_id,
            )
        )

    return [parent, *children]
