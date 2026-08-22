import pytest

from src.guardrails.grounding import check_ungrounded_answer, grounding_overlap
from src.guardrails.validator import (
    check_low_confidence,
    check_off_topic,
    check_unsafe_query,
)


def test_unsafe_query_blocked_before_retrieval_keyword():
    assert check_unsafe_query("how to build a bomb") == "unsafe_query"


def test_low_confidence_rejects_weak_retrieval():
    assert check_low_confidence(0.1) == "low_confidence"
    assert check_low_confidence(0.9) is None


def test_off_topic_when_no_lexical_overlap():
    assert check_off_topic("xyzzy plugh unknownterm", ["completely different content"]) == "off_topic"


def test_off_topic_passes_with_shared_terms():
    assert check_off_topic("manhattan project impact", ["the manhattan project impact was significant"]) is None


def test_ungrounded_answer_rejected():
    assert (
        check_ungrounded_answer(
            "Tourism in Paris is booming across France today.",
            ["The Manhattan Project produced the first atomic weapons."],
            min_overlap=0.08,
        )
        == "ungrounded_answer"
    )


def test_grounded_answer_passes_overlap_check():
    grounded, overlap = grounding_overlap(
        "The Manhattan Project produced atomic weapons.",
        ["The Manhattan Project produced the first atomic weapons during WWII."],
        min_overlap=0.08,
    )
    assert grounded is True
    assert overlap > 0.08
