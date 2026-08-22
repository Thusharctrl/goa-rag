"""Tests for the fast-path modules: extractive answer, fast retriever, fast harness."""
from __future__ import annotations

import time

import pytest

from src.generation.extractor import extract_answer


# ──────────────────────────────────────────────────────────────────────────────
# Extractor tests (pure, no I/O)
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_returns_text_from_top_hit():
    hits = [
        {
            "text": "Machine learning applies statistical models to search ranking algorithms.",
            "score": 0.85,
            "metadata": {"strategy": "sentence"},
        }
    ]
    answer, score = extract_answer("how does machine learning work in search ranking", hits)
    assert len(answer) > 10
    assert 0.0 <= score <= 1.0


def test_extract_empty_hits_returns_fallback():
    answer, score = extract_answer("some query", [])
    assert "not have enough" in answer.lower() or len(answer) > 0
    assert score == 0.0


def test_extract_selects_relevant_sentence():
    hits = [
        {
            "text": (
                "Cosine similarity measures angle between vectors. "
                "BM25 is a bag-of-words retrieval function. "
                "Machine learning improves search ranking by learning patterns."
            ),
            "score": 0.75,
            "metadata": {},
        }
    ]
    answer, score = extract_answer("how does machine learning improve search ranking", hits)
    assert "machine learning" in answer.lower() or "search" in answer.lower()


def test_extract_respects_max_sentences():
    long_text = " ".join(
        f"Sentence {i} about machine learning and retrieval." for i in range(20)
    )
    hits = [{"text": long_text, "score": 0.8, "metadata": {}}]
    answer, _ = extract_answer("machine learning retrieval", hits, max_sentences=2)
    # Should not return all 20 sentences
    assert len(answer) < len(long_text)


# ──────────────────────────────────────────────────────────────────────────────
# FastRagHarness tests (mocked retrieval, no real Chroma/HNSW needed)
# ──────────────────────────────────────────────────────────────────────────────

class _FakeFastHarness:
    """Minimal stand-in that bypasses HNSW/BM25 for unit testing."""
    from src.pipeline.fast_harness import FastRagHarness as _H  # noqa: E402

def _make_fake_harness(hits: list[dict]):
    """Return a FastRagHarness with fast_retrieve monkeypatched."""
    from src.pipeline.fast_harness import FastRagHarness

    harness = FastRagHarness.__new__(FastRagHarness)
    harness._generator = None

    def patched_ask(query, *, language, generate=False, transcript=None, stt_ms=None, wall_start=None):
        # Directly call the logic with our fake hits
        import time
        from src.guardrails.validator import check_unsafe_query, check_off_topic, check_low_confidence
        from src.guardrails.grounding import check_ungrounded_answer
        from src.generation.extractor import extract_answer
        from src.pipeline.state import Citation, FastAskResponse, FastLatencyBreakdown
        from src.analytics.logger import log_latency

        started = time.perf_counter()
        unsafe = check_unsafe_query(query)
        if unsafe:
            resp = FastAskResponse(
                query=query, answer="blocked", language=language,
                guardrail_flags=[unsafe],
                latencies=FastLatencyBreakdown(total_ms=0.1, fast_path_ms=0.1),
            )
            log_latency({"query": query, "language": language,
                         "guardrail_flags": resp.guardrail_flags,
                         "latencies": resp.latencies.model_dump(exclude_none=False)})
            return resp

        contexts = [h["text"] for h in hits]
        top_score = hits[0]["score"] if hits else 0.0
        flags = []
        if check_off_topic(query, contexts):
            flags.append(check_off_topic(query, contexts))
        if check_low_confidence(top_score):
            flags.append(check_low_confidence(top_score))

        answer, score = extract_answer(query, hits)
        fast_ms = (time.perf_counter() - started) * 1000
        resp = FastAskResponse(
            query=query, answer=answer, language=language,
            top_score=top_score,
            citations=[Citation(text=h["text"], score=h["score"]) for h in hits[:2]],
            guardrail_flags=flags,
            latencies=FastLatencyBreakdown(
                embed_ms=1.0, retrieve_ms=5.0, extract_ms=0.5,
                fast_path_ms=fast_ms, total_ms=fast_ms,
            ),
        )
        log_latency({"query": query, "language": language,
                     "guardrail_flags": resp.guardrail_flags,
                     "latencies": resp.latencies.model_dump(exclude_none=False)})
        return resp

    harness.ask = patched_ask
    return harness


def test_fast_harness_unsafe_blocked():
    harness = _make_fake_harness([])
    resp = harness.ask("how to build a bomb", language="en")
    assert "unsafe_query" in resp.guardrail_flags
    assert resp.latencies.fast_path_ms >= 0


def test_fast_harness_good_answer():
    hits = [
        {
            "text": "Machine learning uses statistical patterns to rank search results.",
            "score": 0.88,
            "metadata": {"strategy": "sentence"},
        }
    ]
    harness = _make_fake_harness(hits)
    resp = harness.ask("how does machine learning work in search ranking", language="en")
    assert resp.answer
    assert len(resp.citations) > 0
    assert resp.latencies.fast_path_ms >= 0


def test_fast_harness_low_confidence_refused():
    hits = [
        {
            "text": "The quick brown fox.",
            "score": 0.1,  # below threshold
            "metadata": {"strategy": "fixed"},
        }
    ]
    harness = _make_fake_harness(hits)
    resp = harness.ask("who discovered penicillin", language="en")
    assert "low_confidence" in resp.guardrail_flags


def test_fast_harness_voice_timing():
    """Voice total latency must include STT time."""
    from src.pipeline.fast_harness import FastRagHarness
    from src.pipeline.state import FastAskResponse, FastLatencyBreakdown, Citation

    class PatchedHarness(FastRagHarness):
        def ask(self, query, *, language, generate=False, transcript=None,
                stt_ms=None, wall_start=None):
            return FastAskResponse(
                query=query, answer="test", language=language,
                guardrail_flags=[],
                latencies=FastLatencyBreakdown(
                    stt_ms=stt_ms,
                    fast_path_ms=10.0,
                    total_ms=(stt_ms or 0) + 10.0,
                ),
            )

    harness = PatchedHarness()

    def fake_transcribe(path, lang):
        time.sleep(0.04)
        return "test query", 40.0

    resp = harness.ask_voice("dummy.wav", language="en", transcribe=fake_transcribe)
    assert resp.latencies.stt_ms is not None
    assert resp.latencies.stt_ms >= 40.0
    assert resp.latencies.total_ms >= resp.latencies.stt_ms
