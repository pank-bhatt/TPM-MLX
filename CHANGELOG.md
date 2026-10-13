# Changelog

All notable changes to **TPM-MLX** are documented in this file.

---

## [0.5.0] - 2026-10-13

### 🎨 Qwen-Image 2.1 (Turbo 8-Step & Base 40-Step) Integration
- **Native Qwen-Image 2.1 Support (`MLXImageEngine`):** Integrated both `OsaurusAI/Qwen-Image-2.1-Turbo-mflux-4bit` (custom 8-step `qwen_turbo` non-uniform sigma scheduler with `use_terminal=False`) and `mlx-community/Qwen-Image-2.1-MLX-4bit` (40-step linear flow-matching pipeline).
- **Transparent `format: mlx` Safetensors Weight Adapter:** Added automatic in-memory key remapping, modulation/timestep dequantization, and 3D patch projection transposition so `mlx-community` Qwen-Image-2.1 checkpoints load natively into `QwenImage21Edit`.
- **Unified Text-to-Image & Instruction Editing Pipeline:** Powered by Qwen3-VL 8B multimodal conditioning with prefix KV caching (`use_kv_cache=True`), running in **9.43 GB** unified memory for generation and **10.26 GB** for `1024×1024` instruction editing (`4.7×` speedup with Turbo).
- **Interactive Denoising Steps Slider & Local Cache Filtering:** Added a dynamic steps slider in the Web Playground (auto-defaulting to `8` for Qwen Turbo, `40` for Qwen Base, and `4` for FLUX/Z-Image/Bonsai) and ensured `/v1/models` exclusively lists locally downloaded checkpoints.

### 🧬 OpenAI-Compatible `/v1/embeddings` Endpoint (EmbeddingGemma 2)
- **Sovereign Embedding Engine (`MLXEmbeddingEngine`):** Added `POST /v1/embeddings` powered by `mlx-community/embeddinggemma-2-4bit` (4-bit Gemma 4 architecture) with Matryoshka Representation Learning (MRL) dimension truncation (`dimensions`) and unit L2 re-normalization.

### ♻️ Mutual Unified Memory Eviction & Live Active Memory Telemetry
- **Automatic Cross-Engine Eviction:** Switching between LLM/VLM text models and Diffusion image models automatically evicts the inactive engine (`unload_text_engine()`, `unload_image_engine()`, `unload_embedding_engine()`) and flushes Metal GPU caches (`mx.clear_cache()`, `mx.reset_peak_memory()`).
- **Real-Time Active Memory Reporting:** Added `get_active_memory_gb()` (`mx.get_active_memory()`) across `/v1/models` and Playground metric badges to reflect live allocated unified memory.

### ⚡ Dynamic MTP Auto-Pairing, Gemma 4 QAT Fix & VLM Local Path Resolution
- **Dynamic MTP Companion Auto-Pairing:** Automatically discovers and pairs matching MTP heads and assistant drafters for Qwen 3.x and Gemma 4 models, with cross-family mismatch guards and resilient background model loading.
- **Gemma 4 QAT Assistant Dequantization Fix:** Resolved quantized weight loading for official Gemma 4 QAT assistant drafters.
- **VLM Local Disk Image Resolution:** Resolves `/images/` and `/images/uploads/` references directly from disk during multimodal `/v1/chat/completions` requests to prevent loopback HTTP deadlocks, while preserving `tool_calls` and `tool_call_id` metadata.

### 🏗️ Clean Modular Architecture & Unified Benchmark Suite
- **Layered Backend & Modular ES Frontend:** Decoupled FastAPI server into `routes/`, `schemas/`, `state.py`, `director.py`, and `cache.py`, and split the Web Playground into modular ES modules and CSS with persistent multi-session chat history and responsive mobile navigation.
- **Consolidated Benchmark Hub (`benchmarks/`):** Organized all benchmarks into [`benchmarks/text/`](benchmarks/text/README.md) (13-model LLM/MTP matrix) and [`benchmarks/image/`](benchmarks/image/README.md) (24-run, 6-model text-to-image matrix + 3-model `1024×1024` instruction editing showdown), backed by **34/34 passing unit and architecture tests**.

---

## [0.4.0] - 2026-09-18

### 🌲 Native Support for Ternary Bonsai 2 27B (`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`)
- **Fast Walsh-Hadamard Transform (`fwht`) Runtime:** Vectorized Hadamard activation transform (`fwht`) powered by native Apple Silicon Metal `mx.hadamard_transform`.
- **Hardware-Accelerated `Packed` Layers:** Dynamically unpacks and executes 402 Hadamard-rotated 2-bit affine ternary layers ($\{-s, 0, +s\}$) via `mx.quantized_matmul`.
- **Multimodal & Reasoning Capabilities:** Full support for streaming generation, reasoning tokens (`<think>` blocks), and visual reasoning (Qwen3VL processor) at **20.8 TPS** in **~8.6 GB VRAM**.
- **Model Categorization Safeguards:** Correctly categorizes Bonsai 2 27B as a multimodal LLM in `get_cached_models()`, preventing confusion with diffusion image models.

### ⚡ Direct Mode & Configurable AI Visual Director
- **Zero-Distillation Direct Mode (`auto_expand: false`):** Bypasses text LLM distillation entirely for raw prompt generation, dropping distillation latency from ~4.6s to `< 0.1s` and speeding up 512×512 generation by **25.6%**.
- **Playground UI Toggle:** Added interactive `🪄 AI Visual Director` toggle switch in the Web Playground with dynamic status descriptions.
- **Lean Server Default:** Switched default server text companion to `mlx-community/gemma-4-e2b-it-4bit` + MTP assistant drafter, dropping text VRAM footprint by **47%** (to 2.8 GB at 132.6 TPS).
- **Pure Image Mode (`--no-llm`):** Added CLI `--no-llm` flag and dynamic `POST /v1/load_model {"model": "none"}` memory purging to reclaim text memory on demand.

### 🎨 Z-Image Turbo & FLUX.2 Empirical Telemetry
- **Z-Image Turbo Integration:** Official support for `justintime47/Z-Image-Turbo-MLX-Serve-8bit` offering high-fidelity 8-bit generation in ~14–16s.
- **Isolated Memory Tracking:** Integrated `mx.reset_peak_memory()` before generation to report precise isolated run memory instead of cumulative process lifetime memory.
- **Dynamic Model-Aware Playground:** UI dynamically adapts to active model capabilities (e.g. `✏️ Modify Image` active for FLUX and disabled for non-editing models, inline button feedback for `Copy Link`).

### 🔒 Security Hardening & Pre-Publish Code Quality
- **Bug Fixes:** Resolved 4 critical issues (unbound `mx` NameError, `Sequence` typing import, eager eval on `text_encoder_2`, memory leak on image model unload).
- **Path Traversal Protection:** Hardened input image path resolution against directory traversal.
- **Automated Verification:** Added unit test suite for image engine, uploads, edits, and Fast Walsh-Hadamard numerical invertibility (19/19 tests passing).

---

## [0.3.0] - 2026-09-08

### 🎨 Native Apple Silicon Image Generation & Editing Engine
- **Dedicated Diffusion Engine (`MLXImageEngine`):** High-performance text-to-image and image-to-image editing powered natively by Apple Silicon MLX flow-matching kernels.
- **FLUX.2 & Bonsai Support:** First-class support for `mlx-community/flux2-klein-9b-4bit`, `flux2-klein-4b`, and 2-bit ternary `prism-ml/bonsai-image-ternary-4B-mlx-2bit`.
- **OpenAI-Compatible Endpoints:** Added `/v1/images/generations` and instruction-based `/v1/images/edits` endpoints with real-time telemetry (generation time, VRAM peak, seed, step latency).
- **Intelligent Prompt Distillation:** Conversational `/image` prompts automatically distill complex visual directives into optimized diffusion prompts using local LLM reasoning.
- **Canvas Resolution & Aspect Ratio Selector:** Web playground sidebar dropdown supporting `1024x1024` (Native Studio HD default), `16:9` (1024x576), `9:16` (576x1024), `4:3` (1024x768), `3:4` (768x1024), and draft `512x512`, alongside smart prompt tag extraction.
- **Robustness & Pipe Hardening:** Integrated `SafeStream` to prevent background task crashes on disconnected stdout/stderr pipes, and auto-retry logic for model loading.
- **UI Reasoning Toggle Fix:** Resolved DOM double-toggle conflict with dedicated state badge indicators (`[ON]` / `[OFF]`).

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
- Added automated reproducibility script `benchmarks/text/run_full_matrix.py` and published comprehensive Apple Silicon benchmark report in [`benchmarks/text/README.md`](benchmarks/text/README.md) across 13 model configurations.
- 100% test pass rate across all unit tests in `benchmarks/test_engine.py` and `benchmarks/test_mtp.py`.

---

## [0.1.0] - 2026-08-25

### Initial Release
- **Pre-allocated Static KV Cache (`PreAllocatedKVCache`):** Pre-allocated contiguous memory buffers avoiding dynamic fragmentation.
- **Zero CPU-GPU Sync Autoregressive Loop:** Concurrent GPU evaluation using `mx.eval(token, cache)`.
- **Live Reasoning Tag Parser:** Streaming state-machine normalizer for `<think>...</think>` and `<|channel>thought...<channel|>`.
- **FastAPI OpenAI-Compatible Server & Web Playground:** Interactive browser interface on port 2505.
