# Changelog

All notable changes to **TPM-MLX** are documented in this file.

---

## [0.2.0] - 2026-08-28

### 🚀 Multi-Token Prediction (MTP) Self-Speculation Engine
- **Native Transformer Trunk MTP Support:** Implemented direct sequential Multi-Token Prediction execution for models with native MTP heads (Qwen 3.x, Xiaomi MiMo, DeepSeek-V3).
- **True Trunk Hidden State Routing:** Added `_model_forward()` extraction that feeds the true backbone hidden state into the MTP head, raising token acceptance rates to **62%** and boosting throughput on **Qwen 3.8-27B to 25.46 TPS (+61% speedup)**.
- **Hybrid Linear/Attention Cache Rollback:** Added stateful recurrent snapshotting (`_snapshot_recurrent`) and prefix token replay for hybrid attention architectures (`ArraysCache`).
- **Dynamic Weight Remapping & QuantizedLinear Support:** Automatically extracts, remaps, and un-flattens weights from standalone MTP checkpoints (e.g. `mlx-community/Qwen3.8-27B-MTP-4bit`) with automatic affine 4-bit `QuantizedLinear` conversion.

### 🔀 Zero-Config Dual-Backend Auto-Routing (`mlx-lm` + `mlx-vlm`)
- **Automatic Model Type Detection:** At load time, `MLXEngine` inspects model architecture metadata to automatically route:
  - **Pure Text LLMs** (Qwen, LLaMA, DeepSeek, Mistral) $\to$ **`mlx-lm`** with native trunk MTP heads and pre-allocated KV caches.
  - **Multimodal & Gemma 4 Models** $\to$ **`mlx-vlm`** with official Gemma MTP drafters and vision pipeline.
- **Gemma 4 MTP Drafter Integration:** Seamlessly attaches official assistant checkpoints (e.g. `gemma-4-E4B-it-assistant-bf16`, `E2B-assistant`), reaching **132.6 TPS** on Gemma 4 E2B and **77.0 TPS** on Gemma 4 E4B.

### 🖼️ OpenAI-Compatible Multimodal Vision Support
- **Multimodal `/v1/chat/completions`:** Supports OpenAI-format image payloads (base64 `data:image/...` and HTTP URLs) alongside text prompts.
- **Backend & Telemetry Reporting:** Added `backend: "llm" | "vlm"`, `speculation_mode: "mtp" | "draft" | "none"`, and `has_mtp: true | false` to `/v1/models` and `/v1/load_model`.

### 📊 Full 13-Model Hardware Benchmark Matrix
- Added automated reproducibility script `benchmarks/run_full_matrix.py` and published comprehensive Apple Silicon benchmark report in [`BENCHMARKS.md`](BENCHMARKS.md) across 13 model configurations.
- 100% test pass rate across all unit tests in `benchmarks/test_engine.py` and `benchmarks/test_mtp.py`.

---

## [0.1.0] - 2026-08-25

### Initial Release
- **Pre-allocated Static KV Cache (`PreAllocatedKVCache`):** Pre-allocated contiguous memory buffers avoiding dynamic fragmentation.
- **Zero CPU-GPU Sync Autoregressive Loop:** Concurrent GPU evaluation using `mx.eval(token, cache)`.
- **Live Reasoning Tag Parser:** Streaming state-machine normalizer for `<think>...</think>` and `<|channel>thought...<channel|>`.
- **FastAPI OpenAI-Compatible Server & Web Playground:** Interactive browser interface on port 2505.
