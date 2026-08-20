#!/usr/bin/env python3
"""Build the local Chroma index from a small MSMARCO-XI validation slice."""

from __future__ import annotations

import argparse

from src.chunking.fixed import chunk_fixed
from src.chunking.parent_child import chunk_parent_child
from src.chunking.sentence import chunk_sentences
from src.config import settings
from src.embeddings.index import VectorIndex
from src.ingestion.loader import load_records


def build_index(sample_size: int | None = None, reset: bool = False) -> None:
    records = load_records(sample_size=sample_size)
    index = VectorIndex()

    if reset:
        index.reset()

    all_chunks = []
    for record in records:
        for passage_index, passage in enumerate(record.passages):
            all_chunks.extend(
                chunk_sentences(
                    passage,
                    strategy="sentence",
                    language=record.language,
                    query_id=record.query_id,
                    passage_index=passage_index,
                )
            )
            all_chunks.extend(
                chunk_fixed(
                    passage,
                    strategy="fixed",
                    language=record.language,
                    query_id=record.query_id,
                    passage_index=passage_index,
                )
            )
            all_chunks.extend(
                chunk_parent_child(
                    passage,
                    language=record.language,
                    query_id=record.query_id,
                    passage_index=passage_index,
                )
            )

    batch_size = 128
    for start in range(0, len(all_chunks), batch_size):
        index.upsert_chunks(all_chunks[start : start + batch_size])

    print(
        f"Indexed {len(all_chunks)} chunks from {len(records)} records "
        f"into {settings.chroma_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the voice RAG vector index")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.sample_size,
        help="Number of MSMARCO-XI rows to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Chroma collection",
    )
    args = parser.parse_args()
    build_index(sample_size=args.sample_size, reset=args.reset)


if __name__ == "__main__":
    main()
