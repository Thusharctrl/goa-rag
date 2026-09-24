# Voice RAG — Low-Latency Multilingual RAG

A FastAPI-based Retrieval-Augmented Generation (RAG) system that supports **text and voice queries in English and Hindi**, with a dedicated **low-latency fast path** designed around a sub-200 ms response target.

The system combines:

* multilingual sentence embeddings
* ONNX Runtime for fast CPU inference
* in-memory HNSW dense retrieval
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)
* extractive answers for the latency-critical path
* optional Groq LLM polishing
* Sarvam AI speech-to-text for voice input
* lightweight safety, relevance, and grounding guardrails
* per-stage latency instrumentation and percentile reporting
* a minimal browser UI for text and voice interaction

> **Important:** The repository contains the application code, but not the generated `data/` index/cache or a benchmark-result file. Latency depends on hardware, dataset size, warm-up state, and external API/network conditions. The `<200 ms` target applies to the **text fast path with `generate=false`**; optional LLM generation and voice transcription add external network latency.

---

## Architecture

### Fast text path

```text
User Query
    │
    ▼
FastRagHarness
    │
    ├── Input safety check
    │
    ▼
ONNX multilingual embedding
    │
    ├───────────────┐
    ▼               ▼
HNSW dense       BM25 lexical
retrieval        retrieval
    │               │
    └───────┬───────┘
            ▼
      RRF rank fusion
            │
            ▼
   Top-k deduplication
            │
            ▼
  Relevance guardrails
  • off-topic check
  • low-confidence check
            │
            ▼
 Extractive answer selection
            │
            ▼
   Grounding validation
            │
            ▼
 Answer + citations + latency
            │
            └───────► optional Groq polish
```

### Voice path

```text
Microphone
    │
    ▼
Browser records audio
    │
    ▼
FastAPI voice endpoint
    │
    ▼
Sarvam AI STT (saaras:v3)
    │
    ▼
Transcript
    │
    ▼
Fast text RAG path
```

The production-oriented fast path keeps expensive work off the hot request path wherever possible:

* the embedding session is cached
* HNSW is loaded into RAM once at startup
* BM25 indexes are built once per language
* heavy objects are not recreated for every request
* the default answer path does **not** call an LLM
* the LLM is only initialized when `generate=true`

---

## What the system uses

| Layer             | Technology                              | Role                                                               |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------ |
| API               | FastAPI + Uvicorn                       | HTTP API and application server                                    |
| Validation        | Pydantic / pydantic-settings            | Request/response schemas and configuration                         |
| Dataset           | Hugging Face `datasets`                 | Streaming ingestion of MSMARCO-XI parquet data                     |
| Vector store      | ChromaDB                                | Persistent local index used during indexing and by the legacy path |
| Embeddings        | `paraphrase-multilingual-MiniLM-L12-v2` | English/Hindi semantic representations                             |
| Fast inference    | ONNX Runtime                            | Cached CPU embedding inference                                     |
| Dense retrieval   | hnswlib                                 | In-memory cosine HNSW search                                       |
| Lexical retrieval | rank-bm25                               | Per-language BM25 retrieval                                        |
| Rank fusion       | Reciprocal Rank Fusion                  | Combines dense and lexical rankings                                |
| Fast answering    | Custom extractive selector              | Returns relevant source sentences without an LLM                   |
| Generation        | Groq API                                | Optional grounded answer polishing                                 |
| Speech-to-text    | Sarvam AI `saaras:v3`                   | English/Hindi voice transcription                                  |
| Retry handling    | Tenacity                                | Retries transient Groq generation failures                         |
| Metrics           | JSONL + custom percentile code          | Latency logging and P50/P70/P100 summaries                         |
| Frontend          | Plain HTML/CSS/JavaScript               | Browser demo UI                                                    |
| Testing           | Pytest                                  | Unit and pipeline behavior tests                                   |

### Primary vs. legacy components

The current `/ask` fast path uses:

`ONNX → HNSW + BM25 → RRF → extractive answer → grounding checks`

The legacy `/ask-text` path uses:

`SentenceTransformers → Chroma → lightweight lexical rerank → Groq generation → grounding check`

A few helper modules such as `src/retrieval/bm25.py`, `src/retrieval/fusion.py`, and `src/retrieval/reranker.py` support the older/auxiliary retrieval design and are not all part of the main low-latency path.

---

## Key design decisions

### 1. ONNX embeddings instead of per-request PyTorch inference

The configured embedding model is:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The fast path exports the Hugging Face model to ONNX on first use and caches the resulting model under:

```text
data/onnx_cache/
```

The ONNX session is cached for the lifetime of the process and uses:

* CPU execution
* up to 4 intra-op threads
* `ORT_ENABLE_ALL` graph optimizations
* maximum sequence length of 128
* mean pooling over the last hidden state
* L2-normalized embeddings

Dynamic INT8 weight quantization is attempted when the required ONNX tooling is available; otherwise the code falls back to FP32.

### 2. Chroma for persistence, HNSW for the hot path

ChromaDB is used as the persistent local vector store during indexing.

At application startup, the fast retriever:

1. loads all vectors and metadata from Chroma
2. builds an in-memory HNSW graph
3. builds language-specific BM25 indexes
4. keeps those structures cached in RAM

After warm-up, normal fast-path requests do not query Chroma directly.

### 3. Hybrid retrieval

Two retrieval signals are combined:

**Dense retrieval**

* HNSW
* cosine similarity
* language filtering
* candidate over-fetching before filtering

**Lexical retrieval**

* BM25Okapi
* whitespace tokenization after lowercasing
* separate index per language

The rankings are combined with **Reciprocal Rank Fusion (RRF)** using `k=60`.

The RRF score is used for ordering. The system deliberately uses the **top dense cosine similarity** as its confidence signal because RRF scores are on a different scale.

### 4. Extractive answering for the latency-critical path

With `generate=false`, the system does not call a language model after retrieval.

Instead, `extract_answer()`:

1. splits retrieved passages into sentences
2. removes a lightweight English/Hindi stopword list
3. computes query-term overlap
4. blends sentence relevance with passage score
5. returns up to three highly relevant sentences

This keeps answer generation local and extremely lightweight.

### 5. Optional LLM polish

Setting:

```json
{
  "generate": true
}
```

adds a Groq call after the fast path.

The prompt instructs the model to:

* answer only from retrieved context
* respond in the requested language
* say when the context is insufficient
* avoid unsupported claims

Groq latency is measured separately as `llm_ms`.

The default model is:

```text
openai/gpt-oss-20b
```

### 6. Guardrails

The system applies multiple cheap checks:

**Pre-retrieval safety check**

Blocks queries containing configured unsafe keywords such as:

* bomb / explosive
* kill / suicide / terror
* hack / malware / ransomware

**Post-retrieval relevance checks**

* `off_topic`: retrieved context has no meaningful lexical overlap with the query
* `low_confidence`: top dense similarity is below `MIN_RETRIEVAL_SCORE`

**Output grounding check**

The final answer is compared with retrieved context using token overlap. Answers below the configured grounding threshold are replaced by a verification failure message.

These are intentionally lightweight heuristics so they do not consume the latency budget like an additional model call would.

---

## Chunking

The indexing pipeline creates chunks using three strategies for every passage:

| Strategy       | Behavior                                            | Parameters                          |
| -------------- | --------------------------------------------------- | ----------------------------------- |
| Sentence-aware | Groups sentences while staying around a target size | `target_chars=256`                  |
| Fixed window   | Sliding character windows                           | `512` chars with `64` chars overlap |
| Parent-child   | Full passage parent plus sentence-level children    | parent/child IDs                    |

Chunk IDs include language, row, query, passage, strategy, and chunk index so that chunks remain globally identifiable.

The fast retriever deduplicates results using `parent_id` / `chunk_id` where available.

---

## Dataset

The index builder streams:

```text
ai4bharat/MSMARCO-XI
validation/hinval.parquet
```

The loader creates both:

* English records
* Hindi records

from each source row when the required fields are available.

Default configuration:

```text
SAMPLE_SIZE=100
```

Larger samples can be indexed by changing `SAMPLE_SIZE` or passing `--sample-size`.

Because the generated `data/` directory is ignored by Git and is not included in this ZIP, the index must be built locally before starting the fully warmed application.

---

## Project structure

```text
voice-rag/
├── backend/
│   └── main.py                    # FastAPI application and API routes
│
├── frontend/
│   └── index.html                 # Browser UI for text + voice
│
├── scripts/
│   ├── build_index.py             # Dataset ingestion + chunking + Chroma indexing
│   └── benchmark_latency.py       # Fast-path and LLM latency benchmark
│
├── src/
│   ├── analytics/
│   │   ├── logger.py              # JSONL latency logging
│   │   └── percentiles.py         # P50/P70/P100 metrics
│   │
│   ├── chunking/
│   │   ├── base.py
│   │   ├── fixed.py
│   │   ├── sentence.py
│   │   └── parent_child.py
│   │
│   ├── embeddings/
│   │   ├── model.py               # SentenceTransformers legacy embedder
│   │   ├── onnx_model.py          # Fast ONNX embedder
│   │   └── index.py               # Persistent Chroma vector index
│   │
│   ├── generation/
│   │   ├── extractor.py           # Fast extractive answer selection
│   │   └── generator.py           # Optional Groq generation
│   │
│   ├── guardrails/
│   │   ├── validator.py           # Safety / topic / confidence checks
│   │   └── grounding.py           # Answer grounding validation
│   │
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   └── loader.py
│   │
│   ├── pipeline/
│   │   ├── fast_harness.py        # Main low-latency orchestration
│   │   ├── harness.py             # Legacy pipeline
│   │   └── state.py               # API models and latency schemas
│   │
│   ├── retrieval/
│   │   ├── fast_retriever.py      # Hybrid fast retrieval utilities
│   │   ├── hnsw_index.py          # In-memory HNSW
│   │   ├── bm25_retriever.py      # BM25 over in-memory corpus
│   │   ├── vector.py              # Legacy vector retriever
│   │   ├── reranker.py             # Legacy lightweight reranker
│   │   ├── bm25.py                 # Auxiliary BM25 helper
│   │   └── fusion.py               # Auxiliary RRF helper
│   │
│   └── stt/
│       └── sarvam_client.py       # Sarvam speech-to-text client
│
├── tests/
│   ├── test_chunking.py
│   ├── test_fast_path.py
│   ├── test_guardrails.py
│   └── test_harness.py
│
├── check_duplicates.py             # Chunk ID duplicate checker
├── test_latency.py                 # Manual API latency smoke test
├── verify_voice.py                 # Voice-path verification helper
├── .env.example                    # Environment variable template
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Requirements

Recommended baseline:

* Python **3.10+**
* pip
* internet access during first model/data download
* enough local disk and RAM for the embedding model, Chroma corpus, HNSW graph, and BM25 structures

The repository was packaged with a Python 3.12 virtual-environment skeleton, but that environment does not contain the project's runtime dependencies. Use a fresh environment and install `requirements.txt`.

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repository-url>
cd goa-rag-main/voice-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key

SAMPLE_SIZE=100
CHROMA_PATH=data/chroma
GROQ_MODEL=openai/gpt-oss-20b
```

### Which keys are required?

| Feature                          | Required key     |
| -------------------------------- | ---------------- |
| `/ask` with `generate=false`     | None             |
| `/ask` with `generate=true`      | `GROQ_API_KEY`   |
| `/ask-text` legacy generation    | `GROQ_API_KEY`   |
| `/ask-voice` / `/ask-voice-fast` | `SARVAM_API_KEY` |
| Voice + `generate=true`          | Both keys        |

---

## Build the index

Before the first normal query, build the persistent Chroma index:

```bash
python -m scripts.build_index
```

To rebuild from scratch:

```bash
python -m scripts.build_index --reset
```

To index a larger dataset slice:

```bash
python -m scripts.build_index --sample-size 1000
```

The resulting files are stored under:

```text
data/chroma/
```

The fast path later loads those vectors into RAM and builds HNSW + BM25 at startup.

---

## Run the application

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

For local development with auto-reload:

```bash
uvicorn backend.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

The root route serves the included browser UI.

---

## API

### `GET /health`

Simple health check.

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

---

### `POST /ask`

Primary low-latency text endpoint.

Request:

```json
{
  "query": "What is reciprocal rank fusion?",
  "language": "en",
  "generate": false
}
```

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is reciprocal rank fusion?",
    "language": "en",
    "generate": false
  }'
```

Set `generate=true` to optionally polish the retrieved answer with Groq.

---

### `POST /ask-voice-fast`

Voice → Sarvam STT → fast RAG.

Form fields:

* `audio`: uploaded audio file
* `language`: `en` or `hi`
* `generate`: `true` or `false`

Example:

```bash
curl -X POST http://localhost:8000/ask-voice-fast \
  -F "audio=@test.wav" \
  -F "language=en" \
  -F "generate=false"
```

---

### `POST /ask-voice`

Voice endpoint also used by the included browser UI.

```text
Browser microphone
    → /ask-voice
    → Sarvam STT
    → FastRagHarness
```

The frontend currently sends `generate=false`.

---

### `POST /ask-text`

Legacy text path kept for compatibility.

This path uses the persistent Chroma-backed retriever and Groq generation instead of the optimized extractive fast path.

---

### `GET /metrics`

Returns percentile latency summaries from the JSONL analytics log.

Reported metrics include:

* total latency
* retrieval latency
* generation latency
* fast-path latency
* STT latency
* recent request events

Example:

```bash
curl http://localhost:8000/metrics
```

---

## Response shape

The fast endpoint returns a structure similar to:

```json
{
  "query": "What is reciprocal rank fusion?",
  "transcript": null,
  "answer": "...",
  "language": "en",
  "top_score": 0.82,
  "citations": [
    {
      "text": "...",
      "score": 0.91,
      "strategy": "sentence"
    }
  ],
  "guardrail_flags": [],
  "latencies": {
    "stt_ms": null,
    "embed_ms": 4.1,
    "retrieve_ms": 8.7,
    "extract_ms": 0.5,
    "guardrails_ms": 0.7,
    "fast_path_ms": 14.2,
    "llm_ms": null,
    "total_ms": 14.3
  }
}
```

Actual values depend on the machine, workload, warm-up state, and whether external services are involved.

---

## Latency design

The important distinction is between **fast-path latency** and **end-to-end external-service latency**.

### Fast path

For `POST /ask` with:

```json
"generate": false
```

the measured path is approximately:

```text
guardrails
+ ONNX embedding
+ HNSW
+ BM25
+ RRF
+ deduplication
+ extractive answer
+ grounding
```

The code exposes individual timings so the 200 ms target can be investigated by stage rather than by a single opaque number.

### Optional LLM path

```text
fast path + Groq network/API latency
```

This can easily exceed the 200 ms target because the external LLM call dominates.

### Voice path

```text
Sarvam STT network latency
+ fast path
```

Speech transcription is therefore a separate latency budget from retrieval and answering.

---

## Benchmarking

Warm the in-memory indexes and run the repository's benchmark script:

```bash
python -m scripts.benchmark_latency
```

Save the benchmark JSON to a custom path:

```bash
python -m scripts.benchmark_latency \
  --output data/analytics/benchmark_latest.json
```

The benchmark reports:

* all fast-path requests
* successful fast-path answers
* guardrail-refused requests
* optional LLM-polish requests
* P50, P70, and P100 latency

### Interpreting the benchmark

For the core task, focus primarily on:

```text
fast_path_ms
```

with:

```text
generate=false
```

Do not use LLM-polish or voice end-to-end latency as the direct measure of the local RAG fast-path target.

---

## Testing

The repository currently contains **19 pytest test cases** covering:

* chunk ID uniqueness
* extractive answer selection
* empty/weak retrieval behavior
* fast-path orchestration
* unsafe-query blocking
* low-confidence refusal
* off-topic detection
* answer grounding
* voice latency accounting
* latency percentile calculation

Run:

```bash
python -m pytest -v
```

Compile-check the Python source without executing it:

```bash
python -m compileall -q src backend scripts tests
```

Additional manual checks:

```bash
python test_latency.py
```

```bash
python verify_voice.py
```

`verify_voice.py` requires a working `SARVAM_API_KEY`.

---

## Analytics and latency logs

Requests are written as JSON Lines to:

```text
data/analytics/latency.jsonl
```

Each event stores:

* query
* language
* guardrail flags
* detailed latency breakdown

`GET /metrics` reads this file and computes:

```text
P50
P70
P100
count
```

This design makes it easy to inspect latency distributions instead of relying on one timing sample.

---

## Startup behavior

The application deliberately spends work during startup to make request-time latency predictable.

Startup warm-up includes:

1. ONNX export if the cached model does not exist
2. ONNX Runtime session creation
3. HNSW graph construction from Chroma
4. BM25 index construction
5. loading the legacy SentenceTransformers model

This means the first application startup can be significantly slower than subsequent queries.

A restart therefore has a startup cost, but request handling can remain much faster after warm-up.

---

## Configuration

The main settings live in `src/config.py`.

| Setting                 |                                                       Default | Purpose                            |
| ----------------------- | ------------------------------------------------------------: | ---------------------------------- |
| `SAMPLE_SIZE`           |                                                         `100` | Dataset rows used by index builder |
| `CHROMA_PATH`           |                                                 `data/chroma` | Persistent Chroma storage          |
| `collection_name`       |                                                  `msmarco_xi` | Chroma collection                  |
| `embedding_model`       | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model       |
| `GROQ_MODEL`            |                                          `openai/gpt-oss-20b` | Optional LLM model                 |
| `RETRIEVAL_TOP_K`       |                                                           `5` | Final number of retrieved passages |
| `MIN_RETRIEVAL_SCORE`   |                                                        `0.55` | Dense-score confidence threshold   |
| `MIN_GROUNDING_OVERLAP` |                                                        `0.08` | Answer grounding threshold         |
| `ANALYTICS_LOG_PATH`    |                                `data/analytics/latency.jsonl` | Latency event log                  |
| `SARVAM_API_KEY`        |                                                         empty | Voice STT authentication           |

Most settings are configurable through environment variables because the application uses `pydantic-settings`.

---

## Browser UI

The included frontend is a single HTML/CSS/JavaScript page with:

* English/Hindi language selector
* microphone recording
* text query box
* pipeline progress display
* transcript display for voice requests
* answer display
* retrieved citations with relevance scores
* per-stage latency metrics
* raw JSON developer details

No frontend build tool is required.

The page is served directly by FastAPI from:

```text
frontend/index.html
```

---

## Troubleshooting

### `Chroma collection is empty`

Build the index first:

```bash
python -m scripts.build_index
```

### `GROQ_API_KEY is required`

You are using:

```text
generate=true
```

or the legacy `/ask-text` path.

Add a valid `GROQ_API_KEY` to `.env`.

### `SARVAM_API_KEY is required for voice input`

Add a valid `SARVAM_API_KEY` to `.env` before using the voice endpoints.

### First startup is slow

This is expected when ONNX export or model downloads happen for the first time. The application intentionally moves expensive initialization into startup instead of the request path.

### Answers are frequently refused

Check:

1. `SAMPLE_SIZE`
2. actual dataset coverage
3. `MIN_RETRIEVAL_SCORE`
4. the benchmark queries
5. whether the requested topic exists in the indexed corpus

A refusal can be caused by corpus coverage rather than a failure in the retrieval pipeline.

---

## Known limitations

### Dataset coverage

The default 100-row sample is a development-sized corpus, not a production-scale knowledge base. Retrieval quality depends heavily on whether the requested topic exists in the indexed slice.

### Heuristic guardrails

The safety, off-topic, confidence, and grounding checks are deliberately lightweight heuristics. They are useful as fast filters but should not be treated as a complete production safety system.

### Voice latency

Sarvam STT is an external API call, so end-to-end voice latency is not equivalent to local RAG latency.

### LLM latency

Groq generation is an external network dependency and is outside the local fast-path budget.

### HNSW memory scaling

The fast path loads the entire indexed vector corpus into RAM. This is effective for a moderate local corpus, but memory requirements grow with corpus size.

### Dataset/index rebuild

Changing the sample size or source data requires rebuilding the persistent Chroma index.

---

## Performance tuning ideas

For larger-scale or stricter latency targets, the current design can be extended with:

* batch ingestion and index-build parallelism
* smaller or quantized embedding models
* more aggressive ONNX graph optimization
* a dedicated ANN/vector service for larger corpora
* pre-tokenized BM25 documents
* retrieval result caching
* query embedding caching
* asynchronous external LLM/STT calls where appropriate
* load testing under concurrent traffic
* a more principled reranker if latency allows
* production-grade telemetry instead of a local JSONL file

The key architectural constraint is to keep the default answer path free from a network LLM call.

---

## Quick start

```bash
# 1. Enter project
cd voice-rag

# 2. Create environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env

# 5. Build local RAG index
python -m scripts.build_index

# 6. Start API
uvicorn backend.main:app --reload --port 8000

# 7. Open UI
# http://localhost:8000

# 8. Run tests
python -m pytest -v

# 9. Run latency benchmark
python -m scripts.benchmark_latency
```

---

## Summary

This project is a **hybrid multilingual RAG system optimized around a local, extractive fast path**.

The central performance strategy is:

```text
Persistent Chroma
      ↓
Startup-time in-memory HNSW + BM25
      ↓
ONNX query embedding
      ↓
Hybrid retrieval + RRF
      ↓
Extractive answer
      ↓
Grounding + guardrails
```

Groq and Sarvam are intentionally kept as optional external stages so the core retrieval-and-answer path can be evaluated independently against the latency target.

---

## License

No license file is included in the provided repository. Add the appropriate license before public distribution.
