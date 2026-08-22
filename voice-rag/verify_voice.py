import time
import wave
import json
from src.pipeline.fast_harness import FastRagHarness
from src.stt.sarvam_client import SarvamSTTClient

# Ensure test.wav exists
with wave.open("test.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(8000)
    f.writeframes(b'\x00' * 16000)

harness = FastRagHarness()
# Ensure warmup
from src.retrieval.fast_retriever import warmup_fast_retriever
warmup_fast_retriever()

stt = SarvamSTTClient()

print("--- Testing ask_voice ---")
resp = harness.ask_voice(
    "test.wav",
    language="en",
    transcribe=stt.transcribe,
    generate=False
)

out = resp.latencies.model_dump()
print(json.dumps(out, indent=2))

stt_ms = out.get("stt_ms", 0) or 0
fast_path_ms = out.get("fast_path_ms", 0) or 0
total_ms = out.get("total_ms", 0) or 0

print("Verification:")
print(f"fast_path_ms = {fast_path_ms:.2f}")
print(f"expected embed + retrieve + extract + guardrails = {(out.get('embed_ms') or 0) + (out.get('retrieve_ms') or 0) + (out.get('extract_ms') or 0) + (out.get('guardrails_ms') or 0):.2f}")
print(f"total_ms = {total_ms:.2f}, expected stt_ms + fast_path_ms = {stt_ms + fast_path_ms:.2f}")

print("\n--- Testing ask text ---")
resp2 = harness.ask(
    "What is the capital of Goa?",
    language="en",
    generate=False
)
out2 = resp2.latencies.model_dump()
print(json.dumps(out2, indent=2))
print("Verification Text:")
print(f"total_ms = {out2.get('total_ms', 0):.2f}, fast_path_ms = {out2.get('fast_path_ms', 0):.2f}")
