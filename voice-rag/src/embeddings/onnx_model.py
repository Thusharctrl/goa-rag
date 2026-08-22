"""Fast ONNX-based multilingual embedder.

Exports the cached MiniLM-L12-v2 to ONNX (fp32) on first run, then caches
the session for the lifetime of the process.  Int8 dynamic quantization is
applied if the `onnx` package is available; otherwise falls back to fp32.

Cold export (first ever run)  : ~20–40 s (one-time)
Warm inference per query      : 3–8 ms
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.config import settings

_ONNX_DIR = Path("data/onnx_cache")
_MODEL_NAME = settings.embedding_model
_MAX_LEN = 128


def _onnx_path() -> Path:
    slug = hashlib.md5(_MODEL_NAME.encode()).hexdigest()[:8]
    return _ONNX_DIR / f"minilm_{slug}.onnx"


def _export_to_onnx(onnx_file: Path) -> None:
    """Export the HF model to ONNX (fp32). Quantize to int8 if onnx package present."""
    import torch
    from transformers import AutoModel

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = onnx_file.with_suffix(".fp32.onnx")

    print(f"[onnx] Exporting {_MODEL_NAME} → {fp32_path} …")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME)
    model.eval()

    dummy = tokenizer(
        "warm up", return_tensors="pt",
        padding="max_length", truncation=True, max_length=_MAX_LEN,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    # token_type_ids is optional for many multilingual models
    has_tti = "token_type_ids" in dummy
    token_type_ids = dummy.get("token_type_ids", torch.zeros_like(input_ids))

    input_names = ["input_ids", "attention_mask"]
    dynamic = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}

    if has_tti:
        input_names.append("token_type_ids")
        dynamic["token_type_ids"] = {0: "batch", 1: "seq"}

    dynamic["last_hidden_state"] = {0: "batch", 1: "seq"}

    model_inputs = (input_ids, attention_mask) if not has_tti else (input_ids, attention_mask, token_type_ids)

    with torch.no_grad():
        torch.onnx.export(
            model,
            model_inputs,
            str(fp32_path),
            input_names=input_names,
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes=dynamic,
            opset_version=14,
        )
    print(f"[onnx] fp32 export done: {fp32_path.stat().st_size / 1e6:.1f} MB")

    # Try quantization; skip gracefully if onnx package is broken/unavailable
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"[onnx] Quantizing to int8 → {onnx_file} …")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(onnx_file),
            weight_type=QuantType.QInt8,
        )
        fp32_path.unlink(missing_ok=True)
        print(f"[onnx] Int8 done: {onnx_file.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"[onnx] Quantization skipped ({e}); using fp32.")
        fp32_path.rename(onnx_file)


@lru_cache(maxsize=1)
def _get_onnx_session():
    import onnxruntime as ort

    onnx_file = _onnx_path()
    if not onnx_file.exists():
        _export_to_onnx(onnx_file)

    so = ort.SessionOptions()
    so.intra_op_num_threads = min(4, os.cpu_count() or 1)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        str(onnx_file),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    input_names = {inp.name for inp in sess.get_inputs()}
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return tokenizer, sess, input_names


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., np.newaxis].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    count = mask.sum(axis=1).clip(min=1e-9)
    return summed / count


def onnx_embed(texts: list[str]) -> list[list[float]]:
    """Return L2-normalised embeddings using the cached ONNX session."""
    if not texts:
        return []

    tokenizer, sess, input_names = _get_onnx_session()
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=_MAX_LEN,
        return_tensors="np",
    )
    feeds: dict[str, np.ndarray] = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = enc.get(
            "token_type_ids", np.zeros_like(enc["input_ids"])
        ).astype(np.int64)

    outputs = sess.run(["last_hidden_state"], feeds)
    pooled = _mean_pool(outputs[0], enc["attention_mask"])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    pooled = pooled / norms
    return pooled.tolist()


def warmup_onnx() -> None:
    """Force ONNX export (if needed) and prime the inference session."""
    t0 = time.perf_counter()
    onnx_embed(["warmup sentence"])
    print(f"[onnx] Warmup done in {(time.perf_counter()-t0)*1000:.0f} ms")
