import re

from src.config import settings
from src.embeddings.model import embed_texts


_UNSAFE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(bomb|explosive|kill|suicide|terror)\b",
        r"\b(hack|malware|ransomware)\b",
    ]
]


def check_unsafe_query(query: str) -> str | None:
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query):
            return "unsafe_query"
    return None


def check_low_confidence(top_score: float) -> str | None:
    if top_score < settings.min_retrieval_score:
        return "low_confidence"
    return None


def check_off_topic(query: str, contexts: list[str]) -> str | None:
    if not contexts:
        return "off_topic"

    query_vec = embed_texts([query])[0]
    context_vec = embed_texts([" ".join(contexts[:3])])[0]
    similarity = sum(a * b for a, b in zip(query_vec, context_vec))
    if similarity < 0.15:
        return "off_topic"
    return None
