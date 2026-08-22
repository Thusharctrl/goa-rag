import re


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > 2}


def grounding_overlap(answer: str, contexts: list[str], min_overlap: float) -> tuple[bool, float]:
    if not answer.strip() or not contexts:
        return False, 0.0

    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts))
    if not answer_tokens:
        return False, 0.0

    overlap = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
    return overlap >= min_overlap, overlap


def check_ungrounded_answer(answer: str, contexts: list[str], min_overlap: float) -> str | None:
    grounded, _ = grounding_overlap(answer, contexts, min_overlap)
    if not grounded:
        return "ungrounded_answer"
    return None
