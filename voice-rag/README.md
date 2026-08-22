# Voice-Enabled RAG (HH Goa 2026 Task 2)

Voice-enabled retrieval-augmented generation over **ai4bharat/MSMARCO-XI** with English and Hindi support, Sarvam STT, Groq generation, ChromaDB retrieval, guardrails, and latency analytics.

## What is implemented

```text
Text/Voice Input
  -> Sarvam STT (voice only)
  -> unsafe-query guardrail
  -> vector retrieval (ChromaDB)
  -> low-confidence + lexical off-topic guardrails
  -> Groq answer generation
  -> grounding guardrail
  -> structured JSON response + latency logging
```

This is a demo MVP built on a **small Hindi validation slice** of MSMARCO-XI (`validation/hinval.parquet`, default `SAMPLE_SIZE=100`). It is not a full-corpus production deployment.

## Latency expectations

Observed end-to-end latency is typically **~1.5–3 seconds** for text queries and higher for voice because STT is included. This project does **not** meet a `<200 ms` target. Latency is measured per stage and logged to `data/analytics/latency.jsonl`; `/metrics` reports **P50 / P70 / P100** from those events.

Voice `total_ms` is measured from the start of `/ask-voice` and therefore includes **STT + guardrails + retrieval + generation**.

## Chunking strategies

All three strategies index the same MSMARCO-XI passages into one Chroma collection:

1. **Sentence-aware**
   Splits on sentence boundaries (`.`, `?`, `!`, Hindi danda `।`) and merges sentences up to ~256 characters.
   Why: keeps clauses intact so retrieval returns semantically coherent spans.

2. **Fixed sliding window**
   Uses a 512-character window with 64-character overlap.
   Why: catches facts that span sentence boundaries or appear mid-passage where sentence chunking would split them awkwardly.

3. **Parent-child**
   Stores the full passage as a parent chunk plus smaller sentence-derived child chunks linked by `parent_id`.
   Why: retrieval can match precise child spans while the parent preserves broader context for answer generation.

These strategies are complementary: sentence chunks improve readability, fixed windows improve recall across boundaries, and parent-child balances precision with context.

## Dataset loading note

The Hugging Face card suggests:

```python
load_dataset("ai4bharat/MSMARCO-XI", "hi")
```

In practice the published parquet repo exposes only a `default` config, and the bundled loader script expects `.jsonl` files that are not present. Loading `default` can pull the wrong multi-GB language shard.

This project instead streams only:

`validation/hinval.parquet`

via explicit parquet `data_files`, then stops after `SAMPLE_SIZE` rows.

## Setup

```bash
cd voice-rag
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set in `.env`:

```env
GROQ_API_KEY=...
SARVAM_API_KEY=...
SAMPLE_SIZE=100
```

## Build the index once

Only needed if `data/chroma/` is missing or you intentionally want to rebuild the small validation sample:

```bash
HF_HUB_OFFLINE=1 python scripts/build_index.py --sample-size 100 --reset
```

Do **not** rebuild against the full training parquet during demo prep unless you explicitly intend to scale up.

## Run the API and demo UI

```bash
HF_HUB_OFFLINE=1 python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/`.

## API

### `GET /health`

Health check.

### `POST /ask-text`

```json
{
  "query": "what was the immediate impact of the manhattan project",
  "language": "en"
}
```

### `POST /ask-voice`

Multipart form:
- `audio`: audio file
- `language`: `en` or `hi`

### `GET /metrics`

Returns latency percentiles **P50**, **P70**, and **P100** for:
- `total_ms`
- `retrieve_ms`
- `generate_ms`
- `guardrails_ms`
- `stt_ms` (voice requests only; null text events are excluded)

## Guardrails

Current checks are intentionally lightweight:

- **Unsafe query**: keyword blocklist; rejected before retrieval
- **Low-confidence retrieval**: top vector score below threshold; rejected before Groq generation
- **Off-topic**: lexical overlap between query terms and retrieved passages; rejected before generation
- **Ungrounded answer**: token overlap between generated answer and retrieved context; rejected after generation

## Benchmark and tests

```bash
pytest -q
HF_HUB_OFFLINE=1 python scripts/benchmark_latency.py
```

The benchmark writes `data/analytics/benchmark_latest.json` and appends latency events used by `/metrics`.

## Project layout

```text
voice-rag/
  backend/main.py
  scripts/build_index.py
  scripts/benchmark_latency.py
  frontend/index.html
  tests/
  src/
    ingestion/
    chunking/
    embeddings/
    retrieval/
    generation/
    guardrails/
    pipeline/
    analytics/
    stt/
```

## Demo checklist

1. Ensure `data/chroma/` exists from the small validation build
2. Start FastAPI
3. Ask an English text question via `/ask-text`
4. Ask a Hindi text question via `/ask-text`
5. Record or upload audio via the demo UI for `/ask-voice`
6. Run `scripts/benchmark_latency.py` across multiple queries
7. Inspect `/metrics` for P50/P70/P100

## Known limitations

- Small local validation index, not full MSMARCO-XI training data
- End-to-end latency is dominated by embedding load, vector search, and Groq/STT network calls
- Off-topic detection is lexical, not embedding-based
- No semantic chunking model is used
- Live deployment requires hosting the FastAPI app and providing valid Groq/Sarvam API keys

## License

Uses third-party APIs (Groq, Sarvam) and the MSMARCO-XI dataset. Refer to each provider's terms before deployment.
