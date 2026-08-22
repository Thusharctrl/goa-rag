import json
from pathlib import Path

from src.config import settings


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if percentile >= 100:
        return sorted_values[-1]
    index = int(round((percentile / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[index]


def _load_metric_values(metric: str) -> list[float]:
    log_file = settings.analytics_log_file
    if not log_file.exists():
        return []

    values: list[float] = []
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            latencies = payload.get("latencies", {})
            if metric not in latencies:
                continue
            value = latencies[metric]
            if value is None:
                continue
            values.append(float(value))
    return values


def compute_latency_percentiles(metric: str = "total_ms") -> dict[str, float]:
    values = _load_metric_values(metric)
    return {
        "p50": _percentile(values, 50),
        "p70": _percentile(values, 70),
        "p100": _percentile(values, 100),
        "count": float(len(values)),
    }


def load_recent_events(limit: int = 20) -> list[dict]:
    log_file = settings.analytics_log_file
    if not log_file.exists():
        return []

    lines = log_file.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return events[-limit:]
