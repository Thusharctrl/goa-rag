import re
from typing import Iterable


_WHITESPACE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    return _WHITESPACE.sub(" ", text)


def non_empty(values: Iterable[object]) -> list[str]:
    cleaned = [clean_text(item) for item in values]
    return [item for item in cleaned if item]
