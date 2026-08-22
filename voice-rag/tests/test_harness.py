import time

from src.analytics.percentiles import compute_latency_percentiles
from src.pipeline.harness import RagHarness


class FakeRetriever:
    def retrieve(self, query: str, *, language: str, top_k: int) -> list[dict]:
        return [
            {
                "text": "The Manhattan Project produced the first atomic weapons.",
                "score": 0.91,
                "metadata": {"strategy": "sentence", "parent_id": "1:0:parent"},
            }
        ]


class FakeGenerator:
    def generate(self, query: str, contexts: list[str], language: str) -> str:
        return "The Manhattan Project produced atomic weapons."


def test_unsafe_query_does_not_call_retriever(monkeypatch):
    calls = {"retrieve": 0}

    class TrackingRetriever(FakeRetriever):
        def retrieve(self, query: str, *, language: str, top_k: int) -> list[dict]:
            calls["retrieve"] += 1
            return super().retrieve(query, language=language, top_k=top_k)

    harness = RagHarness(retriever=TrackingRetriever(), generator=FakeGenerator())
    response = harness.ask_text("how to build a bomb", language="en")

    assert response.guardrail_flags == ["unsafe_query"]
    assert calls["retrieve"] == 0
    assert response.latencies.generate_ms == 0.0
    assert response.latencies.total_ms >= response.latencies.guardrails_ms


def test_low_confidence_skips_generation():
    class WeakRetriever:
        def retrieve(self, query: str, *, language: str, top_k: int) -> list[dict]:
            return [
                {
                    "text": "Unrelated passage about gardening.",
                    "score": 0.05,
                    "metadata": {"strategy": "fixed"},
                }
            ]

    harness = RagHarness(retriever=WeakRetriever(), generator=FakeGenerator())
    response = harness.ask_text("manhattan project impact", language="en")

    assert "low_confidence" in response.guardrail_flags
    assert response.latencies.generate_ms == 0.0


def test_voice_total_includes_stt(monkeypatch):
    harness = RagHarness(retriever=FakeRetriever(), generator=FakeGenerator())

    def fake_transcribe(audio_path: str, language: str) -> tuple[str, float]:
        started = time.perf_counter()
        time.sleep(0.05)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return "manhattan project impact", elapsed_ms

    response = harness.ask_voice("dummy.wav", language="en", transcribe=fake_transcribe)

    assert response.transcript == "manhattan project impact"
    assert response.latencies.stt_ms >= 40.0
    assert response.latencies.total_ms >= response.latencies.stt_ms


def test_percentiles_include_p100_as_max(tmp_path, monkeypatch):
    from src.config import settings

    log_file = tmp_path / "latency.jsonl"
    log_file.write_text(
        "\n".join(
            [
                '{"latencies": {"total_ms": 100.0, "stt_ms": null}}',
                '{"latencies": {"total_ms": 200.0, "stt_ms": 300.0}}',
                '{"latencies": {"total_ms": 400.0, "stt_ms": null}}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "analytics_log_path", str(log_file))

    total = compute_latency_percentiles("total_ms")
    stt = compute_latency_percentiles("stt_ms")

    assert total["p50"] == 200.0
    assert total["p100"] == 400.0
    assert stt["count"] == 1.0
    assert stt["p100"] == 300.0
