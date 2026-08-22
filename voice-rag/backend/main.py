from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.analytics.percentiles import compute_latency_percentiles, load_recent_events
from src.pipeline.fast_harness import FastRagHarness
from src.pipeline.harness import RagHarness
from src.pipeline.state import (
    AskResponse,
    AskTextRequest,
    FastAskRequest,
    FastAskResponse,
)
from src.stt.sarvam_client import SarvamSTTClient

app = FastAPI(title="Voice RAG", version="0.2.0")

# ── Singletons ────────────────────────────────────────────────────────────────
# All heavy objects are built once at module load time (or at startup warmup).
# No model is constructed per-request.
_old_harness: RagHarness = RagHarness()       # legacy Chroma path
_fast_harness: FastRagHarness = FastRagHarness()  # new fast path
_stt_client: SarvamSTTClient | None = None
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"


def _get_stt_client() -> SarvamSTTClient:
    global _stt_client
    if _stt_client is None:
        _stt_client = SarvamSTTClient()
    return _stt_client


# ── Startup warmup ────────────────────────────────────────────────────────────
@app.on_event("startup")
def warmup() -> None:
    """Pre-build ONNX session, in-memory HNSW, and BM25 indexes.
    This trades startup time (~5–15 s) for near-zero first-request overhead.
    """
    from src.retrieval.fast_retriever import warmup_fast_retriever
    warmup_fast_retriever()
    # Also warm the legacy embedding model so /ask-text still has no cold start
    from src.embeddings.model import get_embedding_model
    get_embedding_model()


# ── Static files ──────────────────────────────────────────────────────────────
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return FileResponse(index_file)


# ── Fast path ─────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=FastAskResponse)
def ask(request: FastAskRequest) -> FastAskResponse:
    """Fast RAG endpoint.

    - generate=false (default): ONNX embed → HNSW+BM25+RRF → extractive answer.
      Typical latency: 20–80 ms warm.
    - generate=true: same fast path, then Groq LLM polishes the answer.
      LLM latency is reported separately in latencies.llm_ms.
    """
    try:
        return _fast_harness.ask(
            request.query.strip(),
            language=request.language,
            generate=request.generate,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-voice-fast", response_model=FastAskResponse)
async def ask_voice_fast(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    generate: bool = Form(default=False),
) -> FastAskResponse:
    """Voice → Sarvam STT → fast RAG path (with optional Groq polish)."""
    if language not in {"en", "hi"}:
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        stt = _get_stt_client()
        return _fast_harness.ask_voice(
            temp_path,
            language=language,
            transcribe=stt.transcribe,
            generate=generate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ── Legacy endpoints (unchanged, kept for compatibility) ──────────────────────
@app.post("/ask-text", response_model=AskResponse)
def ask_text(request: AskTextRequest) -> AskResponse:
    try:
        return _old_harness.ask_text(request.query.strip(), language=request.language)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-voice", response_model=FastAskResponse)
async def ask_voice(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    generate: bool = Form(default=False),
) -> FastAskResponse:
    if language not in {"en", "hi"}:
        raise HTTPException(status_code=400, detail="language must be 'en' or 'hi'")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        temp_path = tmp.name

    try:
        stt = _get_stt_client()
        return _fast_harness.ask_voice(
            temp_path,
            language=language,
            transcribe=stt.transcribe,
            generate=generate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics() -> dict:
    return {
        "total_ms": compute_latency_percentiles("total_ms"),
        "retrieve_ms": compute_latency_percentiles("retrieve_ms"),
        "generate_ms": compute_latency_percentiles("generate_ms"),
        "fast_path_ms": compute_latency_percentiles("fast_path_ms"),
        "stt_ms": compute_latency_percentiles("stt_ms"),
        "recent_events": load_recent_events(limit=10),
    }
