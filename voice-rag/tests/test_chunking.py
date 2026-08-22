from src.chunking.base import Chunk
from src.chunking.fixed import chunk_fixed
from src.chunking.sentence import chunk_sentences
from src.chunking.parent_child import chunk_parent_child


def test_chunking_unique_ids():
    text = "This is a test passage. It has multiple sentences! Let's see if it works."
    
    # Run multiple strategies with same query_id but different row_index
    chunks: list[Chunk] = []
    
    chunks.extend(
        chunk_sentences(
            text,
            strategy="sentence",
            language="en",
            query_id=42,
            row_index=0,
            passage_index=0,
        )
    )
    chunks.extend(
        chunk_fixed(
            text,
            strategy="fixed",
            language="en",
            query_id=42,
            row_index=0,
            passage_index=0,
        )
    )
    chunks.extend(
        chunk_parent_child(
            text,
            language="en",
            query_id=42,
            row_index=0,
            passage_index=0,
        )
    )
    
    # Same query_id, different row_index (simulating same query, different dataset row)
    chunks.extend(
        chunk_sentences(
            text,
            strategy="sentence",
            language="en",
            query_id=42,
            row_index=1,
            passage_index=0,
        )
    )
    
    # Check for unique IDs
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be globally unique"
