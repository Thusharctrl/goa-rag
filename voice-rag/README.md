# Voice-Enabled RAG (HH Goa 2026 Task 2)

A minimal voice-enabled retrieval-augmented generation system built on **ai4bharat/MSMARCO-XI** with English + Hindi support, Sarvam STT, Groq generation, ChromaDB retrieval, guardrails, and latency analytics.

## Architecture

```text
Voice/Text Input -> Sarvam STT (voice only) -> Guardrails -> Vector Retrieval (Chroma)
-> Groq Answer Generation -> Grounding Check -> Structured JSON Response + Latency Logs
```

Chunking strategies indexed together:
- **Sentence-aware** chunks (~256 chars)
- **Fixed sliding window** chunks (512 chars, 64 overlap)
- **Parent-child** chunks (full passage parent + sentence children)

## Dataset loading note

The Hugging Face dataset card suggests:

```python
load_dataset("ai4bharat/MSMARCO-XI", "hi")
```

In practice the published parquet repo only exposes a `default` config and the custom loader script expects `.jsonl` files that are not present. This project avoids multi-GB downloads by streaming only:

`validation/hinval.parquet`

via explicit parquet `data_files`, then stopping after `SAMPLE_SIZE` rows (default `100`).

## Setup

```bash
cd voice-rag
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set these in `.env`:

```env
GROQ_API_KEY=...
SARVAM_API_KEY=...
SAMPLE_SIZE=100
```

## Build the index

```bash
python scripts/build_index.py --sample-size 100 --reset
```

This downloads a small validation slice, chunks passages with all three strategies, embeds them with `paraphrase-multilingual-MiniLM-L12-v2`, and stores them in `data/chroma/`.

## Run the API + demo UI

```bash
python -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`.

## API

### `POST /ask-text`

```json
{
  "query": "What is the capital of France?",
  "language": "en"
}
```

### `POST /ask-voice`

Multipart form:
- `audio`: audio file
- `language`: `en` or `hi`

### `GET /metrics`

Returns latency percentiles **P50**, **P70**, and **P100** for `total_ms`, `retrieve_ms`, `generate_ms`, and `stt_ms`.

## Guardrails

- **Unsafe query** keyword blocklist
- **Low-confidence retrieval** when top score is below threshold
- **Off-topic** query/context embedding mismatch
- **Ungrounded answer** token overlap check against retrieved passages

## Project layout

```text
voice-rag/
  backend/main.py          FastAPI app
  scripts/build_index.py   One-time index builder
  frontend/index.html      Minimal demo UI
  src/
    ingestion/loader.py    MSMARCO-XI parquet streaming loader
    chunking/              Sentence, fixed, parent-child strategies
    embeddings/            SentenceTransformers + ChromaDB
    retrieval/             Vector retrieval (+ BM25/fusion stubs)
    generation/            Groq answer generation
    guardrails/            Validation + grounding checks
    pipeline/harness.py    Orchestration, retries, structured I/O
    analytics/             Latency logging + percentiles
    stt/sarvam_client.py   Sarvam speech-to-text
```

## Demo checklist

1. Build the index with `scripts/build_index.py`
2. Start FastAPI with uvicorn
3. Ask an English text question via `/ask-text`
4. Ask a Hindi text question via `/ask-text`
5. Upload or record audio via the demo UI (`/ask-voice`)
6. Inspect `/metrics` for P50/P70/P100 latency analytics

## License

Uses third-party APIs (Groq, Sarvam) and the MSMARCO-XI dataset. Refer to each provider's terms before deployment.
