#!/usr/bin/env python3
"""Run a reproducible latency benchmark against the live RAG harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import settings
from src.embeddings.model import get_embedding_model
from src.pipeline.harness import RagHarness


DEFAULT_QUERIES = [
    ("en", "what was the immediate impact of the manhattan project"),
    ("en", "how does machine learning work in search ranking"),
    ("en", "who discovered penicillin and when"),
    ("hi", "मैनहटन परियोजना की सफलता का तात्कालिक प्रभाव क्या था"),
    ("hi", "खोज रैंकिंग में मशीन लर्निंग कैसे काम करती है"),
    ("en", "what is the weather on mars today"),
    ("en", "how to build a bomb"),
    ("en", "xyzzy completely unrelated nonsense query"),
]


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a simple nearest-rank-style percentile."""
    if not values:
        return 0.0

    sorted_values = sorted(values)

    if percentile_value >= 100:
        return sorted_values[-1]

    index = int(
        round(
            (percentile_value / 100.0)
            * (len(sorted_values) - 1)
        )
    )

    return sorted_values[index]


def summarize(values: list[float]) -> dict[str, float]:
    """Return P50, P70, P100 and sample count."""
    return {
        "p50": percentile(values, 50),
        "p70": percentile(values, 70),
        "p100": percentile(values, 100),
        "count": float(len(values)),
    }


def run_benchmark(
    *,
    output_path: Path,
    queries: list[tuple[str, str]] | None = None,
) -> dict:
    # Warm the embedding model before timing requests so that the
    # one-time model-loading cost does not contaminate benchmark results.
    get_embedding_model()

    harness = RagHarness()
    results = []

    benchmark_queries = queries or DEFAULT_QUERIES

    for language, query in benchmark_queries:
        response = harness.ask_text(
            query,
            language=language,
        )

        results.append(
            {
                "query": query,
                "language": language,
                "guardrail_flags": response.guardrail_flags,
                "latencies": response.latencies.model_dump(
                    exclude_none=False
                ),
            }
        )

    total_values = [
        float(result["latencies"]["total_ms"])
        for result in results
        if result["latencies"].get("total_ms") is not None
    ]

    retrieve_values = [
        float(result["latencies"]["retrieve_ms"])
        for result in results
        if result["latencies"].get("retrieve_ms") is not None
    ]

    generate_values = [
        float(result["latencies"]["generate_ms"])
        for result in results
        if result["latencies"].get("generate_ms") is not None
    ]

    guardrail_values = [
        float(result["latencies"]["guardrails_ms"])
        for result in results
        if result["latencies"].get("guardrails_ms") is not None
    ]

    stt_values = [
        float(result["latencies"]["stt_ms"])
        for result in results
        if result["latencies"].get("stt_ms") is not None
    ]

    summary = {
        "query_count": len(results),
        "total_ms": summarize(total_values),
        "retrieve_ms": summarize(retrieve_values),
        "generate_ms": summarize(generate_values),
        "guardrails_ms": summarize(guardrail_values),
        "stt_ms": summarize(stt_values),
        "results": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RAG latency percentiles"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analytics/benchmark_latest.json"),
        help="Where to write the benchmark summary JSON",
    )

    args = parser.parse_args()

    summary = run_benchmark(
        output_path=args.output,
    )

    print(
        json.dumps(
            {
                "query_count": summary["query_count"],
                "total_ms": summary["total_ms"],
                "retrieve_ms": summary["retrieve_ms"],
                "generate_ms": summary["generate_ms"],
                "guardrails_ms": summary["guardrails_ms"],
                "stt_ms": summary["stt_ms"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()