from pydantic import BaseModel, Field


class AskTextRequest(BaseModel):
    query: str = Field(min_length=1)
    language: str = Field(default="en", pattern="^(en|hi)$")


class Citation(BaseModel):
    text: str
    score: float
    strategy: str = ""


class LatencyBreakdown(BaseModel):
    stt_ms: float | None = None
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_ms: float = 0.0


class AskResponse(BaseModel):
    query: str
    transcript: str | None = None
    answer: str
    language: str
    citations: list[Citation] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    latencies: LatencyBreakdown
