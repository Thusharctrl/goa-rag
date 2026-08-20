import time
from pathlib import Path

from sarvamai import SarvamAI

from src.config import settings


class SarvamSTTClient:
    def __init__(self) -> None:
        if not settings.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required for voice input")
        self._client = SarvamAI(api_subscription_key=settings.sarvam_api_key)

    def transcribe(self, audio_path: str, language: str) -> tuple[str, float]:
        started = time.perf_counter()
        language_code = "hi-IN" if language == "hi" else "en-IN"

        with Path(audio_path).open("rb") as audio_file:
            response = self._client.speech_to_text.transcribe(
                file=audio_file,
                model="saaras:v3",
                mode="transcribe",
                language_code=language_code,
            )

        transcript = getattr(response, "transcript", None) or getattr(response, "text", "") or ""
        elapsed_ms = (time.perf_counter() - started) * 1000
        return transcript.strip(), elapsed_ms
