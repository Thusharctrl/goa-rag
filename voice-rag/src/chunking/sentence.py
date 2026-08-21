import re

from src.chunking.base import Chunk


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def chunk_sentences(
    text: str,
    *,
    strategy: str,
    language: str,
    query_id: int,
    passage_index: int,
    target_chars: int = 256,
) -> list[Chunk]:
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buffer = ""
    chunk_index = 0

    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if buffer and len(candidate) > target_chars:
            chunk_id = f"{language}:{query_id}:{passage_index}:{strategy}:{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=buffer,
                    strategy=strategy,
                    language=language,
                    query_id=query_id,
                    passage_index=passage_index,
                )
            )
            chunk_index += 1
            buffer = sentence
        else:
            buffer = candidate

    if buffer:
        chunk_id = f"{language}:{query_id}:{passage_index}:{strategy}:{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=buffer,
                strategy=strategy,
                language=language,
                query_id=query_id,
                passage_index=passage_index,
            )
        )

    return chunks
