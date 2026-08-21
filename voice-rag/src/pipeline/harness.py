from __future__ import annotations

import time
from typing import Callable

from src.analytics.logger import log_latency
from src.config import settings
from src.generation.generator import AnswerGenerator
from src.guardrails.grounding import check_ungrounded_answer
from src.guardrails.validator import (
    check_low_confidence,
    check_off_topic,
    check_unsafe_query,
)
from src.pipeline.state import AskResponse, Citation, LatencyBreakdown
from src.retrieval.reranker import rerank
from src.retrieval.vector import VectorRetriever


class RagHarness:
    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever or VectorRetriever()
        self._generator = generator

    def _get_generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator()
        return self._generator

    def ask_text(
        self,
        query: str,
        *,
        language: str,
        transcript: str | None = None,
        stt_ms: float | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        flags: list[str] = []
        unsafe = check_unsafe_query(query)

        if unsafe:
            return self._blocked_response(
                query=query,
                language=language,
                transcript=transcript,
                flag=unsafe,
                started=started,
                stt_ms=stt_ms,
                guardrails_ms=(time.perf_counter() - guardrails_started) * 1000,
            )

        retrieve_started = time.perf_counter()
        hits = self._retriever.retrieve(
            query,
            language=language,
            top_k=settings.retrieval_top_k,
        )
        hits = rerank(query, hits, settings.retrieval_top_k)
        retrieve_ms = (time.perf_counter() - retrieve_started) * 1000
        guardrails_started = time.perf_counter()
        contexts = [hit["text"] for hit in hits]
        top_score = hits[0]["score"] if hits else 0.0

        off_topic = check_off_topic(query, contexts)
        if off_topic:
            flags.append(off_topic)
        low_conf = check_low_confidence(top_score)
        if low_conf:
            flags.append(low_conf)

        guardrails_ms = (time.perf_counter() - guardrails_started) * 1000

        if flags:
            return AskResponse(
                query=query,
                transcript=transcript,
                answer="I do not have enough reliable information to answer that question.",
                language=language,
                citations=[
                    Citation(
                        text=hit["text"],
                        score=hit["score"],
                        strategy=str(hit["metadata"].get("strategy", "")),
                    )
                    for hit in hits[:3]
                ],
                guardrail_flags=flags,
                latencies=LatencyBreakdown(
                    stt_ms=stt_ms,
                    retrieve_ms=retrieve_ms,
                    generate_ms=0.0,
                    guardrails_ms=guardrails_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                ),
            )

        generate_started = time.perf_counter()
        answer = self._get_generator().generate(query, contexts, language)
        generate_ms = (time.perf_counter() - generate_started) * 1000

        ungrounded = check_ungrounded_answer(
            answer,
            contexts,
            settings.min_grounding_overlap,
        )
        if ungrounded:
            flags.append(ungrounded)
            answer = "I could not verify this answer against the retrieved passages."

        response = AskResponse(
            query=query,
            transcript=transcript,
            answer=answer,
            language=language,
            citations=[
                Citation(
                    text=hit["text"],
                    score=hit["score"],
                    strategy=str(hit["metadata"].get("strategy", "")),
                )
                for hit in hits
            ],
            guardrail_flags=flags,
            latencies=LatencyBreakdown(
                stt_ms=stt_ms,
                retrieve_ms=retrieve_ms,
                generate_ms=generate_ms,
                guardrails_ms=guardrails_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )

        log_latency(
            {
                "query": query,
                "language": language,
                "guardrail_flags": flags,
                "latencies": response.latencies.model_dump(),
            }
        )
        return response

    def _blocked_response(
        self,
        *,
        query: str,
        language: str,
        transcript: str | None,
        flag: str,
        started: float,
        stt_ms: float | None,
        guardrails_ms: float,
    ) -> AskResponse:
        response = AskResponse(
            query=query,
            transcript=transcript,
            answer="This request cannot be processed for safety reasons.",
            language=language,
            guardrail_flags=[flag],
            latencies=LatencyBreakdown(
                stt_ms=stt_ms,
                guardrails_ms=guardrails_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )
        log_latency(
            {
                "query": query,
                "language": language,
                "guardrail_flags": [flag],
                "latencies": response.latencies.model_dump(),
            }
        )
        return response

    def ask_voice(
        self,
        audio_path: str,
        *,
        language: str,
        transcribe: Callable[[str, str], tuple[str, float]],
    ) -> AskResponse:
        transcript, stt_ms = transcribe(audio_path, language)
        return self.ask_text(
            transcript,
            language=language,
            transcript=transcript,
            stt_ms=stt_ms,
        )
