# TPM-MLX: Optimized Apple Silicon Inference Engine

![TPM-MLX vs Ollama Benchmarks](tpm_mlx_vs_ollama_benchmarks.png)

**TPM-MLX (`tpm`)** is a zero-bloat, high-performance local inference engine designed specifically for Apple Silicon hardware. Powered by Apple’s native `mlx`, `mlx-lm`, and `mlx-vlm` Metal kernels, it achieves state-of-the-art tokens-per-second (TPS) throughput by combining:
- **Native Multi-Token Prediction (MTP) Self-Speculation** (up to **1.61× speedup** on 27B models).
- **Zero-Config Dual-Backend Auto-Routing** (`mlx-lm` for pure LLMs + `mlx-vlm` for multimodal VLMs & Gemma 4).
- **Pre-Allocated Static Key-Value (KV) Caching** (`PreAllocatedKVCache`) to eliminate GPU memory fragmentation.
- **Live Streaming Reasoning Tag Normalizer** (`<think>` and `<|channel>thought`).
- **OpenAI-Compatible REST API + Multimodal Vision Support** with a glassmorphic **Web Playground** on port `2505`.

---

## 🚀 Key Features

### 1. ⚡ Native Multi-Token Prediction (MTP) Engine
* **Transformer Trunk Self-Speculation:** Directly executes sequential MTP prediction heads using internal backbone hidden states for **Qwen 3.x, Xiaomi MiMo, and DeepSeek-V3** with negligible overhead.
* **Recurrent Cache Rollback:** Employs stateful recurrent snapshotting (`_snapshot_recurrent`) and prefix token replay for hybrid linear/attention caches (`ArraysCache`), reaching **62% acceptance rate** and **25.5 TPS (1.61× boost)** on Qwen 3.8-27B.
* **Standalone Checkpoint Auto-Remapping:** Automatically detects, un-flattens, and loads standalone quantized MTP checkpoints (e.g. `mlx-community/Qwen3.8-27B-MTP-4bit`) with dynamic 4-bit `QuantizedLinear` layer conversion.

### 2. 🔀 Zero-Config Dual-Backend Auto-Routing (`mlx-lm` + `mlx-vlm`)
* **Zero User Friction:** Automatically inspects `config.json` at model load time and routes:
  * **Pure Text LLMs** (Qwen, LLaMA, DeepSeek, Mistral) $\to$ **`mlx-lm`** + Native Trunk MTP.
  * **Multimodal & Gemma 4 Models** $\to$ **`mlx-vlm`** + Official Gemma MTP Drafters.
* **Gemma 4 MTP Assistant Drafters:** Seamlessly attaches companion drafters, reaching **132.6 TPS** on Gemma 4 E2B and **77.0 TPS** on Gemma 4 E4B.

### 3. 🧠 Pre-allocated Static KV Cache
* Pre-allocates contiguous memory buffers up to the maximum sequence size (default `4096`) on the first forward pass.
* Eliminates dynamic memory reallocation overhead and OS memory fragmentation during long conversations.

### 4. 🎭 Live Reasoning State Machine Normalizer
* Real-time streaming state-machine parser for `<think>...</think>` and `<|channel>thought...<channel|>` tags.
* Filters reasoning by default (`--no-reasoning`) for immediate answers, or normalizes it for clean rendering in Web UIs and terminal clients.

---

## 📊 Local Benchmarks (Apple Silicon Metal GPU)

| Model | Architecture | Quant | Speculation Mode | Baseline Speed | TPM-MLX Speed | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.8 27B** | 27B Dense (Hybrid Linear/Attn) | 4-bit | **Native MTP Head** | 15.81 TPS | **25.46 TPS** | **1.61× (+61%)** 🚀 |
| **Gemma 4 E2B** | 2B Dense Edge | 4-bit | **Official MTP Drafter** | 115.54 TPS | **132.60 TPS** | **1.15× (+15%)** 🚀 |
| **Gemma 4 E4B** | 4B Dense Edge | 4-bit | **Official MTP Drafter** | 68.17 TPS | **77.00 TPS** | **1.13× (+13%)** 🚀 |
| **Gemma 4 26B-A4B** | 26B MoE (4B Active) | 4-bit | Native MoE Baseline | 68.59 TPS | **81.12 TPS** | **Fastest (Native MoE)** |
| **Qwen 2.5 1.5B** | 1.5B Dense Edge | 4-bit | Standard Autoregressive | 213.15 TPS | **213.15 TPS** | Instant Edge Inference |
| **Gemma 3 1B IT** | 1B Dense Edge | 4-bit | Standard Autoregressive | 247.88 TPS | **247.88 TPS** | Ultra-Fast Edge |

> [!TIP]
> View the complete 13-model benchmark report and hardware performance analysis in [BENCHMARKS.md](BENCHMARKS.md).

---

## 🛠️ Installation

### Prerequisites
* macOS (Apple Silicon M1/M2/M3/M4)
* Python >= 3.12
* [uv](https://github.com/astral-sh/uv) (recommended)

### Quick Setup
```bash
git clone https://github.com/pank-bhatt/TPM-MLX.git
cd TPM-MLX
uv pip install -e .
```

---

## 💻 CLI Usage

### 1. Launch API Server & Web Playground
```bash
# Standard LLM:
uv run tpm serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 2505

# Qwen 3.8 with Native MTP self-speculation:
uv run tpm serve --model mlx-community/Qwen3.8-27B-4bit --draft-model mlx-community/Qwen3.8-27B-MTP-4bit --port 2505

# Gemma 4 with official MTP assistant drafter:
uv run tpm serve --model mlx-community/gemma-4-e4b-it-4bit --draft-model mlx-community/gemma-4-E4B-it-assistant-bf16 --port 2505
```
Open **`http://localhost:2505`** in your browser for the Web Playground.

### 2. Interactive Terminal Chat
```bash
uv run tpm chat --model mlx-community/Qwen3.8-27B-4bit --draft-model mlx-community/Qwen3.8-27B-MTP-4bit
```
* Use `--reasoning` to display internal thought chains.
* Type `/exit` or `/quit` to close.

### 3. Run Benchmarks
```bash
# Single Model Benchmark:
uv run python benchmarks/benchmark.py --model "mlx-community/Qwen3.8-27B-4bit" --draft-model "mlx-community/Qwen3.8-27B-MTP-4bit"

# Full 13-Model Hardware Sanity & Matrix Suite:
uv run python benchmarks/run_full_matrix.py
```

---

## 🔌 OpenAI-Compatible API Endpoints

### `/v1/chat/completions` (POST)
Supports standard OpenAI payloads, SSE streaming, multimodal image inputs, and custom `"reasoning"` toggling.

#### 1. Text Completion Example:
```bash
curl -X POST http://localhost:2505/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.8-27B-4bit",
    "messages": [{"role": "user", "content": "Explain quantum superposition in 2 sentences."}],
    "stream": false,
    "max_tokens": 256
  }'
```

#### 2. Multimodal Vision Example (with image_url):
```bash
curl -X POST http://localhost:2505/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe what is in this image."},
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

#### 3. Model Telemetry (`/v1/models`):
```bash
curl http://localhost:2505/v1/models
```
Returns active model info, `backend` (`"llm"` or `"vlm"`), `speculation_mode` (`"mtp"`, `"draft"`, or `"none"`), and `has_mtp`.

---

## 🔒 Security & Code Quality

* **Safetensors Native**: Loads only `.safetensors` model weights to eliminate arbitrary code execution risks.
* **XSS Sanitized UI**: Markdown renderer escapes HTML and sanitizes URI schemes to prevent script injection.
* **OpenAPI Validation**: Strict boundary schema validation powered by Pydantic and FastAPI.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
