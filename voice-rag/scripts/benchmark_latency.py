#!/usr/bin/env python3
"""Latency benchmark for the Voice RAG pipeline.

Reports three tables:
  A) ALL requests  (fast path, no LLM)
  B) SUCCESSFUL answers that passed all guardrails
  C) REFUSED requests (guardrail-blocked, generate_ms == 0)

And a fourth table:
  D) LLM-polished answers (10 selected queries with generate=True)

Usage:
  python3 -m scripts.benchmark_latency
  python3 -m scripts.benchmark_latency --output data/analytics/benchmark_latest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.fast_harness import FastRagHarness
from src.retrieval.fast_retriever import warmup_fast_retriever

# ──────────────────────────────────────────────────────────────────────────────
# Query corpus (26 fast-path + 10 LLM-polish subset)
# ──────────────────────────────────────────────────────────────────────────────
FAST_QUERIES: list[tuple[str, str]] = [
    # --- English: likely answerable from MSMARCO-XI corpus ---
    ("en", "how does machine learning work in search ranking"),
    ("en", "what is the difference between precision and recall"),
    ("en", "what are retrieval augmented generation systems"),
    ("en", "how does BM25 compare to dense retrieval"),
    ("en", "what is document chunking in RAG pipelines"),
    ("en", "how is cosine similarity used for semantic search"),
    ("en", "what is the MSMARCO dataset"),
    ("en", "what are sentence transformers used for"),
    ("en", "how does top-k retrieval work in vector databases"),
    ("en", "what is reciprocal rank fusion"),
    # --- Hindi: in-domain ---
    ("hi", "खोज रैंकिंग में मशीन लर्निंग कैसे काम करती है"),
    ("hi", "सूचना पुनर्प्राप्ति में प्रेसीजन और रिकॉल क्या है"),
    ("hi", "वेक्टर डेटाबेस में शीर्ष-क retrieval कैसे काम करता है"),
    ("hi", "भाषा मॉडल में tokenization क्या होती है"),
    ("hi", "embedding मॉडल किस प्रकार भाषाओं को समझते हैं"),
    ("hi", "RAG pipeline में chunking क्यों जरूरी है"),
    # --- OOD (expected low_confidence refusal) ---
    ("en", "what was the immediate impact of the manhattan project"),
    ("en", "who discovered penicillin and when"),
    ("en", "what is the weather on mars today"),
    ("en", "who won the FIFA World Cup in 2022"),
    ("en", "what is the capital city of France"),
    # --- Unsafe (pre-retrieval block) ---
    ("en", "how to build a bomb"),
    ("en", "how to make malware"),
    ("en", "how to perform a terror attack"),
    # --- Nonsense ---
    ("en", "xyzzy completely unrelated nonsense query"),
    ("en", "frobzorp quizzle snorfblat undefined"),
]

# Subset for LLM-polish timing (only queries likely to succeed)
LLM_POLISH_QUERIES: list[tuple[str, str]] = [
    ("en", "how does machine learning work in search ranking"),
    ("hi", "खोज रैंकिंग में मशीन लर्निंग कैसे काम करती है"),
    ("en", "what is the MSMARCO dataset"),
    ("hi", "embedding मॉडल किस प्रकार भाषाओं को समझते हैं"),
    ("en", "how does BM25 compare to dense retrieval"),
]


# ──────────────────────────────────────────────────────────────────────────────
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    if p >= 100:
        return sv[-1]
    return sv[int(round((p / 100.0) * (len(sv) - 1)))]


def _summarize(values: list[float]) -> dict:
    return {
        "p50": _pct(values, 50),
        "p70": _pct(values, 70),
        "p100": _pct(values, 100),
        "count": len(values),
    }


def _pick(results: list[dict], key: str, positive_only: bool = False) -> list[float]:
    out: list[float] = []
    for r in results:
        v = r["latencies"].get(key)
        if v is None:
            continue
        f = float(v)
        if positive_only and f <= 0:
            continue
        out.append(f)
    return out


def _is_successful(r: dict) -> bool:
    """An extractive-path result counts as successful if extract_ms > 0 (extract ran)."""
    return float(r["latencies"].get("extract_ms", 0)) > 0


def _section(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "total_ms": _summarize(_pick(rows, "total_ms")),
        "fast_path_ms": _summarize(_pick(rows, "fast_path_ms")),
        "embed_ms": _summarize(_pick(rows, "embed_ms", positive_only=True)),
        "retrieve_ms": _summarize(_pick(rows, "retrieve_ms", positive_only=True)),
        "extract_ms": _summarize(_pick(rows, "extract_ms", positive_only=True)),
        "guardrails_ms": _summarize(_pick(rows, "guardrails_ms")),
    }


# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(*, output_path: Path) -> dict:
    print(f"\nWarming up fast retriever (ONNX + HNSW + BM25)…")
    warmup_fast_retriever()
    print("  Warmup done.\n")

    harness = FastRagHarness()
    fast_results: list[dict] = []

    print(f"Running {len(FAST_QUERIES)} fast-path queries (generate=False)…\n")
    for idx, (lang, query) in enumerate(FAST_QUERIES, 1):
        resp = harness.ask(query, language=lang, generate=False)
        lat = resp.latencies.model_dump(exclude_none=False)
        ok = _is_successful({"latencies": lat})
        status = "✓" if ok else f"✗ ({', '.join(resp.guardrail_flags)})"
        print(
            f"  [{idx:02d}/{len(FAST_QUERIES)}] [{lang}] {query[:55]:<55}  "
            f"fast={lat['fast_path_ms']:6.1f}ms  "
            f"ret={lat.get('retrieve_ms', 0):5.1f}ms  "
            f"{status}"
        )
        fast_results.append({
            "query": query,
            "language": lang,
            "guardrail_flags": resp.guardrail_flags,
            "latencies": lat,
        })

    successful = [r for r in fast_results if _is_successful(r)]
    refused = [r for r in fast_results if not _is_successful(r)]

    # LLM polish timing
    print(f"\nRunning {len(LLM_POLISH_QUERIES)} LLM-polish queries (generate=True)…\n")
    llm_results: list[dict] = []
    for idx, (lang, query) in enumerate(LLM_POLISH_QUERIES, 1):
        resp = harness.ask(query, language=lang, generate=True)
        lat = resp.latencies.model_dump(exclude_none=False)
        print(
            f"  [{idx}/{len(LLM_POLISH_QUERIES)}] [{lang}] {query[:55]:<55}  "
            f"fast={lat['fast_path_ms']:6.1f}ms  "
            f"llm={lat.get('llm_ms', 0) or 0:7.1f}ms  "
            f"total={lat['total_ms']:7.1f}ms"
        )
        llm_results.append({
            "query": query,
            "language": lang,
            "guardrail_flags": resp.guardrail_flags,
            "latencies": lat,
        })

    llm_times = [
        float(r["latencies"].get("llm_ms") or 0)
        for r in llm_results
        if r["latencies"].get("llm_ms") and float(r["latencies"]["llm_ms"]) > 0
    ]

    summary = {
        "query_count": len(fast_results),
        "all": _section(fast_results),
        "successful": _section(successful),
        "refused": _section(refused),
        "llm_polish": {
            "count": len(llm_results),
            "fast_path_ms": _summarize(_pick(llm_results, "fast_path_ms")),
            "llm_ms": _summarize(llm_times),
            "total_ms": _summarize(_pick(llm_results, "total_ms")),
        },
        # Legacy flat keys
        "total_ms": _summarize(_pick(fast_results, "total_ms")),
        "retrieve_ms": _summarize(_pick(fast_results, "retrieve_ms", True)),
        "generate_ms": _summarize(llm_times),
        "guardrails_ms": _summarize(_pick(fast_results, "guardrails_ms")),
        "stt_ms": _summarize(_pick(fast_results, "stt_ms", True)),
        "fast_results": fast_results,
        "llm_results": llm_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _print_section(label: str, section: dict) -> None:
    n = section.get("count", 0)
    if n == 0:
        print(f"\n  {label}: 0 queries")
        return
    print(f"\n  {label} (n={n})")
    for metric in ("fast_path_ms", "total_ms", "embed_ms", "retrieve_ms", "extract_ms", "guardrails_ms", "llm_ms"):
        s = section.get(metric, {})
        if not s or s.get("count", 0) == 0:
            continue
        print(
            f"    {metric:<16}  "
            f"P50={s['p50']:7.1f}ms  "
            f"P70={s['p70']:7.1f}ms  "
            f"P100={s['p100']:7.1f}ms  "
            f"(n={s['count']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG latency percentiles")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analytics/benchmark_latest.json"),
    )
    args = parser.parse_args()

    summary = run_benchmark(output_path=args.output)

    print("\n" + "=" * 80)
    print("BENCHMARK — FAST PATH (generate=False, ONNX + HNSW + BM25 + Extractive)")
    print("=" * 80)
    _print_section("A) ALL requests", summary["all"])
    _print_section("B) SUCCESSFUL (extractive answer returned)", summary["successful"])
    _print_section("C) REFUSED (guardrail blocked)", summary["refused"])
    print("\n" + "-" * 80)
    print("OPTIONAL LLM POLISH (generate=True, Groq openai/gpt-oss-20b)")
    print("-" * 80)
    _print_section("D) LLM-polished", summary["llm_polish"])
    print(f"\n  Saved: {args.output}\n")


if __name__ == "__main__":
    main()