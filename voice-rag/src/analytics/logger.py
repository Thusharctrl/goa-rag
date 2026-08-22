import json
from pathlib import Path

from src.config import settings


def log_latency(payload: dict) -> None:
    log_file: Path = settings.analytics_log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
