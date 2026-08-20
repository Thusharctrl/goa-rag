from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analytics.percentiles import compute_latency_percentiles, load_recent_events
from src.pipeline.harness import RagHarness
from src.pipeline.state import AskResponse, AskTextRequest
from src.stt.sarvam_client import SarvamSTTClient

app = FastAPI(title="Voice RAG", version="0.1.0")
harness = RagHarness()
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return FileResponse(index_file)


@app.post("/ask-text", response_model=AskResponse)
def ask_text(request: AskTextRequest) -> AskResponse:
    try:
        return harness.ask_text(request.query.strip(), language=request.language)
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-voice", response_model=AskResponse)
async def ask_voice(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
) -> AskResponse:
    if language not in {"en", "hi"}:
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await audio.read())
        temp_path = temp_file.name

    try:
        stt_client = SarvamSTTClient()
        return harness.ask_voice(
            temp_path,
            language=language,
            transcribe=stt_client.transcribe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.get("/metrics")
def metrics() -> dict:
    return {
        "total_ms": compute_latency_percentiles("total_ms"),
        "retrieve_ms": compute_latency_percentiles("retrieve_ms"),
        "generate_ms": compute_latency_percentiles("generate_ms"),
        "stt_ms": compute_latency_percentiles("stt_ms"),
        "recent_events": load_recent_events(limit=10),
    }
