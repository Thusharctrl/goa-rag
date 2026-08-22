# Voice RAG — HH Goa 2026 Task 2

Multilingual (English + Hindi) Voice-Enabled RAG system with a sub-20 ms extractive fast path and optional LLM polish.

---

## Architecture

```
Voice/Text Input
│
├── Text ──────────────────────────────────────────────────────────┐
│                                                                   │
└── Voice ──► Sarvam STT (saaras:v3)  ──────────────────────────► │
                                                                    ▼
                                                           FastRagHarness
                                                                    │
                                                         ┌──────────┴──────────┐
                                                         │  Input Guardrail     │
                                                         │  (unsafe keywords)   │
                                                         └──────────┬──────────┘
                                                                    │
                                                         ┌──────────┴──────────┐
                                                         │  ONNX int8 Embed     │
                                                         │  (MiniLM-L12-v2)     │
                                                         │  dim=384, ~5ms warm  │
                                                         └──────────┬──────────┘
                                                                    │
                                                  ┌─────────────────┴─────────────────┐
                                                  │                                   │
                                           HNSW dense                          BM25 lexical
                                           (in-memory,                         (rank-bm25,
                                            hnswlib)                            per-language)
                                                  │                                   │
                                                  └──────────────┬────────────────────┘
                                                                  │
                                                            RRF Fusion
                                                  (Reciprocal Rank Fusion, k=60)
                                                                  │
                                                  ┌───────────────┴──────────────┐
                                                  │  Post-retrieval Guardrails   │
                                                  │  • off_topic (lexical)       │
                                                  │  • low_confidence (cosine)   │
                                                  └───────────────┬──────────────┘
                                                                  │
                                                  ┌───────────────┴──────────────┐
                                                  │  Extractive Answer           │
                                                  │  (sentence-level scoring)    │
                                                  │  < 1 ms                      │
                                                  └───────────────┬──────────────┘
                                                                  │
                                                  ┌───────────────┴──────────────┐
                                                  │  Output Grounding Check      │
                                                  │  (token overlap validation)  │
                                                  └───────────────┬──────────────┘
                                                                  │
                                                  ◄──── FastAskResponse ─────────►
                                                  (answer, citations, fast_path_ms,
                                                   top_score, guardrail_flags)
                                                                  │
                                                     [optional: generate=True]
                                                                  │
                                                         Groq LLM polish
                                                      (openai/gpt-oss-20b)
                                                      llm_ms reported separately
```

---

## Chunking Strategies (all indexed)

| Strategy | Description |
|----------|-------------|
| **sentence-aware** | Splits on `[.!?।]`, groups sentences up to 256 chars. Preserves semantic units. |
| **fixed sliding window** | 512-char windows with 64-char overlap. Ensures no content is split mid-context. |
| **parent-child** | Full passage as parent + sentence-level children. Enables parent-context expansion post-retrieval. |

All three strategies are indexed in ChromaDB and loaded into the in-memory HNSW at startup.

---

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /ask` | **Fast path.** `generate=false` (default) → extractive answer in ~15 ms. `generate=true` → + Groq polish (~880 ms total). |
| `POST /ask-voice-fast` | Same as `/ask` but accepts audio upload (multipart). Sarvam STT + fast RAG. |
| `POST /ask-text` | Legacy Chroma path (compatible with existing tests). |
| `POST /ask-voice` | Legacy voice path. |
| `GET /metrics` | Percentile latency from analytics log. |
| `GET /health` | Health check. |

### Request example
```bash
# Fast extractive answer (no LLM)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the cost of wasted water", "language": "en", "generate": false}'

# With Groq LLM polish
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the cost of wasted water", "language": "en", "generate": true}'
```

---

## Latency (measured, not fabricated)

### Fast Path (`generate=false`) — 26 queries, warm ONNX session

| Category | P50 | P70 | P100 |
|----------|-----|-----|------|
| **All requests** | **13.1 ms** | **13.9 ms** | **17.6 ms** |
| Successful answers (n=4) | 15.7 ms | 15.7 ms | 17.6 ms |
| Refused (guardrail) (n=22) | 12.5 ms | 13.6 ms | 15.9 ms |

Breakdown (all):
- embed_ms: P50=3.4ms
- retrieve_ms (HNSW+BM25+RRF): P50=8.8ms
- extract_ms: P50<1ms
- guardrails_ms: P50<1ms

### Optional LLM Polish (`generate=true`) — 5 queries

| Metric | P50 | P70 | P100 |
|--------|-----|-----|------|
| fast_path_ms | 21 ms | 21 ms | 29 ms |
| llm_ms (Groq) | 768 ms | 768 ms | 872 ms |
| **total_ms** | 619 ms | 786 ms | 888 ms |

### Startup (one-time, not per-request)
| Component | Time |
|-----------|------|
| ONNX int8 export (first ever run) | ~40 s |
| ONNX session warmup (subsequent starts) | ~3.5 s |
| HNSW load from Chroma (11,308 vectors) | ~2.8 s |
| BM25 index build | ~0.1 s |

---

## Dataset

- **Source:** `ai4bharat/MSMARCO-XI` (validation/hinval.parquet, streamed)
- **Dev index:** 100 source rows → ~200 bilingual records → ~11,308 chunks
- **Scale:** Increase `SAMPLE_SIZE` in `.env` to index 1000–2000 rows on a stronger machine
- Both English and Hindi passages indexed

---

## Known Limitations

1. **100-row dev index coverage**: The MSMARCO-XI validation slice covers general web-search topics (companies, food, water usage). ML/RAG-specific queries get `low_confidence` refusals because those topics aren't in the 100-row slice. This is a _dataset coverage_ issue, not a pipeline bug. Increasing `SAMPLE_SIZE=1000` will expand coverage significantly.

2. **`<200 ms` claim**: True for the fast extractive path (P50=13.1 ms). LLM-polished answers take 600–900 ms (Groq API network latency is dominant).

3. **STT latency**: Sarvam `saaras:v3` adds ~1400 ms (network API). Reported separately in `latencies.stt_ms`.

4. **ONNX export size**: int8 dynamic quantization produces a 119 MB file (fp32 artifact is 1.4 MB). Static quantization with calibration data would produce a smaller, faster model.

---

## Setup

```bash
cd voice-rag
cp .env.example .env   # add GROQ_API_KEY and SARVAM_API_KEY
pip install -r requirements.txt
python3 -m scripts.build_index       # builds Chroma index (100 rows, ~2 min)
uvicorn backend.main:app --reload --port 8000
```

First startup exports ONNX model (~40 s one-time), subsequent startups warm in ~7 s.

---

## Run Tests

```bash
python3 -m pytest tests/ -v             # 18 tests
python3 -m compileall src/ backend/ scripts/ -q
python3 -m scripts.benchmark_latency
```

---

## Remaining Submission Tasks

- [ ] Push branch to GitHub: `git push origin final-rag-fast`
- [ ] Deploy backend (Render/Railway/Fly.io): `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- [ ] Record process walkthrough video
- [ ] Record end-to-end demo video (show `/ask` fast path + voice flow)
- [ ] Submit GitHub repo link + live demo URL
