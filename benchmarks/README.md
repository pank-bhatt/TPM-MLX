# TPM-MLX Benchmarks & Test Suite

All hardware performance benchmarks, evaluation harnesses, telemetry artifacts, and visual comparison reports for **TPM-MLX** are organized in this directory.

---

## 📂 Benchmark Categories

### 1. [💬 Text & LLM Speculative Decoding Benchmarks (`benchmarks/text/`)](text/README.md)
- **[`text/README.md`](text/README.md):** Full 13-model LLM & VLM hardware benchmark report across Qwen 3.8, Gemma 4 (E2B, E4B, 12B, 26B-A4B), and Muse Glimmer with Native Multi-Token Prediction (MTP) and official drafters.
- **[`text/tpm_mlx_vs_ollama_benchmarks.png`](text/tpm_mlx_vs_ollama_benchmarks.png):** Throughput (TPS) and Time-to-First-Token (TTFT) comparison chart against Ollama.
- **[`text/benchmark.py`](text/benchmark.py):** Single-model CLI benchmark runner (`TPM-MLX` vs `mlx-lm` baseline).
- **[`text/run_full_matrix.py`](text/run_full_matrix.py) & [`text/matrix_results.json`](text/matrix_results.json):** Automated 13-model matrix harness and telemetry output.
- **[`text/mtp_experiments.py`](text/mtp_experiments.py) & [`text/mtp_experiments_results.json`](text/mtp_experiments_results.json):** Empirical validation across MTP temperature, task domain entropy, and speculation depth ($\gamma = 1, 2, 3$).

### 2. [🎨 Image Generation & Editing Benchmarks (`benchmarks/image/`)](image/README.md)
- **[`image/README.md`](image/README.md):** Comprehensive 6-model text-to-image benchmark (24 runs across `512×512` and `1024×1024`, Direct vs. AI Visual Director) + 3-model instruction-based image editing benchmark (`1024×1024`):
  - **Bonsai 4B** (`2-bit Ternary`, 4 steps)
  - **FLUX.2 Klein 4B** (`Unquantized BF16`, 4 steps)
  - **FLUX.2 Klein 9B** (`4-bit Affine`, 4 steps)
  - **Z-Image Turbo** (`8-bit Affine`, 4 steps)
  - **Qwen-Image 2.1 Turbo** (`4-bit Affine`, 8 steps)
  - **Qwen-Image 2.1 Base** (`4-bit Affine`, 40 steps)
- **[`image/images/`](image/images/):** All 27 benchmark output PNGs.

---

## 🧪 Running Unit & Architecture Tests

```bash
uv run pytest benchmarks/test_architecture.py benchmarks/test_engine.py benchmarks/test_image_engine.py benchmarks/test_mtp.py -v
```
