from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset

from src.config import settings
from src.ingestion.cleaner import clean_text, non_empty


@dataclass(frozen=True)
class RagRecord:
    row_index: int
    query_id: int
    language: str
    query: str
    answer: str
    passages: list[str]
    selected: list[bool]


HINDI_VAL_PARQUET = settings.hf_parquet_url


def _extract_passages(
    example: dict, language: str
) -> tuple[list[str], list[bool]]:
    passages = example.get("passages") or {}

    key = (
        "Translated_passages"
        if language == "hi"
        else "English_passages"
    )

    texts = non_empty(passages.get(key) or [])

    selected = [
        bool(item)
        for item in (passages.get("is_selected") or [])
    ]

    if len(selected) < len(texts):
        selected.extend([False] * (len(texts) - len(selected)))

    return texts, selected[:len(texts)]


def _to_record(example: dict, language: str, row_index: int) -> RagRecord:
    if language == "hi":
        query = clean_text(example.get("query"))
        answer = clean_text(example.get("Answer"))
    else:
        query = clean_text(example.get("Eng_Query"))
        answer = clean_text(example.get("Eng_Answer"))

    passages, selected = _extract_passages(example, language)

    return RagRecord(
        row_index=row_index,
        query_id=int(example.get("query_id") or 0),
        language=language,
        query=query,
        answer=answer,
        passages=passages,
        selected=selected,
    )


def load_records(sample_size: int | None = None) -> list[RagRecord]:
    size = sample_size or settings.sample_size

    dataset = load_dataset(
        "parquet",
        data_files=HINDI_VAL_PARQUET,
        split="train",
        streaming=True,
    )

    records: list[RagRecord] = []

    for index, example in enumerate(dataset):
        if index >= size:
            break

        for language in ("hi", "en"):
            record = _to_record(example, language, index)

            if record.query and record.passages:
                records.append(record)

    return records

def iter_records(sample_size: int | None = None) -> Iterable[RagRecord]:
    yield from load_records(sample_size)