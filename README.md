# TPM-MLX: Optimized Apple Silicon Local Inference Engine

![TPM-MLX vs Ollama Benchmarks](tpm_mlx_vs_ollama_benchmarks.png)

**TPM-MLX (`tpm`)** is a zero-bloat, highly optimized local LLM inference engine built specifically for Apple Silicon hardware. Utilizing direct `mlx` and `mlx-lm` Metal bindings, it achieves maximum tokens-per-second (t/s) throughput by eliminating CPU-GPU synchronization stalls and using a pre-allocated static Key-Value (KV) cache. 

Out-of-the-box, it serves an **OpenAI-compatible REST API**, an **interactive terminal CLI**, and a **premium glassmorphic Web Playground** on port `2505`.

---

## 🚀 Key Performance Enhancements

1. **Pre-allocated Static KV Cache**:
   * Pre-allocates contiguous memory buffers up to the maximum sequence size (default `4096`) on the first forward pass.
   * Enables in-place, zero-copy index updates, avoiding memory fragmentation and dynamic memory reallocation overhead during inference.
   * Falls back to dynamic growth if the prompt exceeds the pre-allocated cache boundary.
2. **Zero CPU-GPU Sync Autoregressive Loop**:
   * Evaluates token indices and KV updates concurrently on the GPU using `mx.eval(token, cache)`. This avoids blocking `.item()` calls, preventing hardware execution stalls.
3. **Dynamic Gemma 4 Configuration Patching**:
   * Detects hybrid global/sliding window attention layers in `gemma4_assistant` models and dynamically overrides parameters (e.g. setting `"num_kv_shared_layers"` to `0`) to bypass sliding window exceptions and KeyError crashes.
   * Uses `strict=False` weight loading to resolve key mismatches in local quantized 4-bit and 8-bit Hugging Face weights.
4. **Reasoning State Machine Parser**:
   * Supports real-time streaming state parsing for both DeepSeek-style (`<think>...</think>`) and Gemma 4-style (`<|channel>thought...<channel|>`) reasoning tags.
   * Intercepts and filters out thinking tokens by default (`--no-reasoning`) for instant answers, or renders them as dimmed terminal text block / collapsible cards in the Web UI when reasoning mode is enabled.

---

## 📊 Local Benchmarks (Apple M4 Pro - 64 GB Unified Memory)

Comparative throughput speed using `mlx-community/gemma-4-e2b-it-4bit` (3B) and `mlx-community/gemma-4-e4b-it-4bit` (10B):

| Model | Engine | Throughput (TPS) | Time-To-First-Token (TTFT) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma-4-e2b-it-4bit` (3B)** | **TPM-MLX (Ours)** | **102.84 t/s** | **125.05 ms** | **PASS** |
| | mlx-lm (Baseline) | 0.00 t/s | 0.00 ms | **FAIL** (Aborted: weight mismatch) |
| **`gemma-4-e4b-it-4bit` (10B)** | **TPM-MLX (Ours)** | **71.51 t/s** | **162.07 ms** | **PASS** |
| | mlx-lm (Baseline) | 0.00 t/s | 0.00 ms | **FAIL** (Aborted: weight mismatch) |

> [!NOTE]
> Standard `mlx-lm` fails to load these checkpoints due to strict parameter checking on hybrid architectures. TPM-MLX dynamically resolves config parameters and bypasses checks, allowing local Apple Silicon chips to hit **100+ t/s**.

> [!TIP]
> For a detailed, task-specific comparison against **Ollama** across 5B, 10B, and 12B architectures (covering simulation, deduction, schema generation, and tool use), see the [BENCHMARKS.md](file:///Users/pank/Experiments/MLX/BENCHMARKS.md) report.

---

## 🛠️ Installation

### Prerequisites
* macOS (Apple Silicon M1/M2/M3/M4)
* Python >= 3.12
* [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Quick Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tpm-mlx.git
   cd tpm-mlx
   ```
2. Install the package in editable mode:
   ```bash
   uv pip install -e .
   ```

---

## 💻 Usage

The engine is controlled via the `tpm` command-line tool.

### 1. Start the API Server & Web Playground
Launch the FastAPI server (default port `2505`):
```bash
uv run tpm serve --model mlx-community/gemma-4-e2b-it-4bit --port 2505
```
Open **`http://localhost:2505`** in your browser to access the Web Playground.

### 2. Run Interactive CLI Chat
Chat with the model directly in your terminal:
```bash
uv run tpm chat --model mlx-community/gemma-4-e2b-it-4bit
```
*   To enable reasoning thoughts printout, run with `--reasoning`.
*   To exit, type `/exit` or `/quit`.

### 3. Pre-download Models
Pre-download any model checkpoint from Hugging Face:
```bash
uv run tpm download mlx-community/gemma-4-e2b-it-4bit
```

### 4. Run the Performance Benchmark Harness
Compare TPM-MLX and Ollama performance across six standard task categories (Simulation, Schema Generation, Logic Deduction, Multi-Turn Dialog, Log Data Needle Extraction, Tool Dispatching):
```bash
uv run python tests/test_perf_harness.py
```
The results table and summary analysis are generated and saved to `tests/perf_report.md`.

---

## 🔌 OpenAI-Compatible API Endpoints

### `/v1/chat/completions` (POST)
Supports standard OpenAI payloads, SSE streaming, and a custom `"reasoning"` toggle parameter.

#### Non-Streaming Example:
```bash
curl -X POST http://localhost:2505/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e2b-it-4bit",
    "messages": [{"role": "user", "content": "Explain Apple Silicon in 1 sentence."}],
    "stream": false,
    "max_tokens": 512,
    "reasoning": false
  }'
```

#### Streaming Example:
```bash
curl -X POST http://localhost:2505/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e2b-it-4bit",
    "messages": [{"role": "user", "content": "Write a short poem."}],
    "stream": true,
    "reasoning": true
  }'
```

---

## 🔒 Security & Code Quality

* **Safetensors Native Weights**: Only loads weights in the standard `.safetensors` format, which is secure against arbitrary pickle-code execution vulnerabilities.
* **XSS Sanitized UI**: The built-in client-side markdown compiler escapes all HTML elements and sanitizes URL schemas inside hyperlink blocks (`[text](url)`) to block `javascript:` and `data:` XSS vectors.
* **OpenAPI Schema Validation**: All REST inputs are strictly validated at the boundaries using FastAPI Pydantic models.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
