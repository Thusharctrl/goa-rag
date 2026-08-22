from fastapi.testclient import TestClient
from backend.main import app
import json
import wave

# Create a small valid WAV file for testing
with wave.open("test.wav", "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(8000)
    f.writeframes(b'\x00' * 16000)

client = TestClient(app)

print("--- Testing /ask-voice-fast ---")
with open("test.wav", "rb") as f:
    resp = client.post(
        "/ask-voice-fast",
        files={"audio": ("test.wav", f, "audio/wav")},
        data={"language": "en"}
    )
    
if resp.status_code == 200:
    data = resp.json()
    latencies = data.get("latencies", {})
    print(json.dumps(latencies, indent=2))
else:
    print(resp.text)

print("\n--- Testing /ask ---")
resp2 = client.post(
    "/ask",
    json={"query": "test query", "language": "en"}
)
if resp2.status_code == 200:
    print(json.dumps(resp2.json().get("latencies", {}), indent=2))
else:
    print(resp2.text)
