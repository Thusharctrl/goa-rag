from pydantic import BaseModel, Field


class AskTextRequest(BaseModel):
    query: str = Field(min_length=1)
    language: str = Field(default="en", pattern="^(en|hi)$")


class FastAskRequest(BaseModel):
    query: str = Field(min_length=1)
    language: str = Field(default="en", pattern="^(en|hi)$")
    generate: bool = Field(default=False, description="If true, also call Groq LLM to polish the answer")


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


class FastLatencyBreakdown(BaseModel):
    stt_ms: float | None = None
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0       # HNSW + BM25 + fusion
    extract_ms: float = 0.0        # extractive answer selection
    guardrails_ms: float = 0.0
    fast_path_ms: float = 0.0      # end-to-end before optional LLM
    llm_ms: float | None = None    # optional Groq polish latency
    total_ms: float = 0.0


class AskResponse(BaseModel):
    query: str
    transcript: str | None = None
    answer: str
    language: str
    citations: list[Citation] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    latencies: LatencyBreakdown


class FastAskResponse(BaseModel):
    query: str
    transcript: str | None = None
    answer: str
    language: str
    top_score: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    latencies: FastLatencyBreakdown

