"""Fast RAG harness using in-memory HNSW + BM25 + extractive answer.

Hot path:
  query → ONNX embed → HNSW + BM25 + RRF → extractive answer → response
  (no LLM call)

Optional polish (generate=True):
  Same hot path, then call Groq to rewrite the extractive answer.
  LLM latency measured and reported separately.
"""
from __future__ import annotations

import time
from typing import Callable

from src.analytics.logger import log_latency
from src.config import settings
from src.generation.extractor import extract_answer
from src.guardrails.grounding import check_ungrounded_answer
from src.guardrails.validator import (
    check_low_confidence,
    check_off_topic,
    check_unsafe_query,
)
from src.pipeline.state import Citation, FastAskResponse, FastLatencyBreakdown
from src.retrieval.fast_retriever import fast_retrieve


class FastRagHarness:
    """Orchestrates the low-latency RAG fast path."""

    def __init__(self) -> None:
        # Generator is created lazily only when `generate=True`
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            from src.generation.generator import AnswerGenerator
            self._generator = AnswerGenerator()
        return self._generator

    # ──────────────────────────────────────────────────────────────────────
    def ask(
        self,
        query: str,
        *,
        language: str,
        generate: bool = False,
        transcript: str | None = None,
        stt_ms: float | None = None,
        wall_start: float | None = None,
    ) -> FastAskResponse:
        started = wall_start if wall_start is not None else time.perf_counter()

        # 1. Input guardrail — unsafe keyword check (pre-retrieval)
        g0 = time.perf_counter()
        unsafe = check_unsafe_query(query)
        guardrails_ms = (time.perf_counter() - g0) * 1000
        if unsafe:
            return self._finalize(FastAskResponse(
                query=query,
                transcript=transcript,
                answer="This request cannot be processed for safety reasons.",
                language=language,
                guardrail_flags=[unsafe],
                latencies=FastLatencyBreakdown(
                    stt_ms=stt_ms,
                    guardrails_ms=guardrails_ms,
                    fast_path_ms=(time.perf_counter() - started) * 1000,
                    total_ms=(time.perf_counter() - started) * 1000,
                ),
            ))

        # 2. Fast retrieval: ONNX embed + HNSW + BM25 + RRF fusion
        e0 = time.perf_counter()
        # embed_ms is dominated by ONNX inference (~3–8 ms warm)
        from src.embeddings.onnx_model import onnx_embed
        query_vec = onnx_embed([query])[0]
        embed_ms = (time.perf_counter() - e0) * 1000

        r0 = time.perf_counter()
        from src.retrieval.hnsw_index import hnsw_query
        from src.retrieval.bm25_retriever import bm25_query
        dense_hits = hnsw_query(query_vec, language=language, top_k=settings.retrieval_top_k * 3)
        lexical_hits = bm25_query(query, language=language, top_k=settings.retrieval_top_k * 3)
        # RRF fusion — used only for ordering
        from src.retrieval.fast_retriever import _rrf
        fused = _rrf([dense_hits, lexical_hits])
        seen: set[str] = set()
        hits: list[dict] = []
        for h in fused:
            meta = h.get("metadata", {})
            key = str(meta.get("parent_id") or meta.get("chunk_id") or h["text"][:80])
            if key not in seen:
                seen.add(key)
                hits.append(h)
            if len(hits) >= settings.retrieval_top_k:
                break
        retrieve_ms = (time.perf_counter() - r0) * 1000

        # Use the dense (cosine) score of the top result for confidence check.
        # RRF scores top out at ~1/61 ≈ 0.016 and cannot be compared to the
        # 0.55 cosine threshold in settings.min_retrieval_score.
        dense_top_score = dense_hits[0]["score"] if dense_hits else 0.0
        # Expose dense score as the canonical confidence signal in the response.
        top_score = dense_top_score
        contexts = [h["text"] for h in hits]

        # 3. Post-retrieval guardrails
        g1 = time.perf_counter()
        flags: list[str] = []
        off = check_off_topic(query, contexts)
        if off:
            flags.append(off)
        low = check_low_confidence(dense_top_score)  # cosine score, not RRF
        if low:
            flags.append(low)
        guardrails_ms += (time.perf_counter() - g1) * 1000

        if flags:
            fast_ms = (time.perf_counter() - started) * 1000
            return self._finalize(FastAskResponse(
                query=query,
                transcript=transcript,
                answer="I do not have enough reliable information to answer that question.",
                language=language,
                top_score=top_score,
                citations=[
                    Citation(
                        text=h["text"],
                        score=h["score"],
                        strategy=str(h["metadata"].get("strategy", "")),
                    )
                    for h in hits[:3]
                ],
                guardrail_flags=flags,
                latencies=FastLatencyBreakdown(
                    stt_ms=stt_ms,
                    embed_ms=embed_ms,
                    retrieve_ms=retrieve_ms,
                    guardrails_ms=guardrails_ms,
                    fast_path_ms=fast_ms,
                    total_ms=fast_ms,
                ),
            ))

        # 4. Extractive answer (fast path, no LLM)
        x0 = time.perf_counter()
        extractive_answer, ext_score = extract_answer(query, hits)
        extract_ms = (time.perf_counter() - x0) * 1000

        fast_ms = (time.perf_counter() - started) * 1000

        # 5. Output grounding check on extractive answer
        g2 = time.perf_counter()
        ungrounded = check_ungrounded_answer(
            extractive_answer, contexts, settings.min_grounding_overlap
        )
        guardrails_ms += (time.perf_counter() - g2) * 1000
        if ungrounded:
            flags.append(ungrounded)
            extractive_answer = "I could not verify this answer against the retrieved passages."

        final_answer = extractive_answer
        llm_ms: float | None = None

        # 6. Optional Groq polish
        if generate and not flags:
            try:
                llm_start = time.perf_counter()
                final_answer = self._get_generator().generate(query, contexts, language)
                llm_ms = (time.perf_counter() - llm_start) * 1000
            except Exception:
                # LLM failure → fall back to extractive answer
                llm_ms = None

        total_ms = (time.perf_counter() - started) * 1000

        return self._finalize(FastAskResponse(
            query=query,
            transcript=transcript,
            answer=final_answer,
            language=language,
            top_score=top_score,
            citations=[
                Citation(
                    text=h["text"],
                    score=h["score"],
                    strategy=str(h["metadata"].get("strategy", "")),
                )
                for h in hits
            ],
            guardrail_flags=flags,
            latencies=FastLatencyBreakdown(
                stt_ms=stt_ms,
                embed_ms=embed_ms,
                retrieve_ms=retrieve_ms,
                extract_ms=extract_ms,
                guardrails_ms=guardrails_ms,
                fast_path_ms=fast_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
            ),
        ))

    def ask_voice(
        self,
        audio_path: str,
        *,
        language: str,
        transcribe: Callable[[str, str], tuple[str, float]],
        generate: bool = False,
    ) -> FastAskResponse:
        started = time.perf_counter()
        transcript, stt_ms = transcribe(audio_path, language)
        return self.ask(
            transcript,
            language=language,
            generate=generate,
            transcript=transcript,
            stt_ms=stt_ms,
            wall_start=started,
        )

    def _finalize(self, response: FastAskResponse) -> FastAskResponse:
        log_latency({
            "query": response.query,
            "language": response.language,
            "guardrail_flags": response.guardrail_flags,
            "latencies": response.latencies.model_dump(exclude_none=False),
        })
        return response
