from src.chunking.base import Chunk


def chunk_fixed(
    text: str,
    *,
    strategy: str,
    language: str,
    query_id: int,
    passage_index: int,
    window_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunk_id = f"{query_id}:{passage_index}:{strategy}:{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    strategy=strategy,
                    language=language,
                    query_id=query_id,
                    passage_index=passage_index,
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
