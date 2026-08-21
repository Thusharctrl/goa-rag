import re

from src.config import settings


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
    """
    Cheap lexical sanity check.

    Retrieval confidence remains the primary signal. This check only
    catches cases where the retrieved text has essentially no overlap
    with the query vocabulary.
    """
    if not contexts:
        return "off_topic"

    query_terms = {
        term.lower().strip(".,!?;:\"'()[]{}")
        for term in query.split()
        if len(term.strip(".,!?;:\"'()[]{}")) >= 3
    }

    if not query_terms:
        return "off_topic"

    context_terms = set()
    for context in contexts[:3]:
        context_terms.update(
            term.lower().strip(".,!?;:\"'()[]{}")
            for term in context.split()
        )

    overlap = query_terms.intersection(context_terms)

    if not overlap:
        return "off_topic"

    return None