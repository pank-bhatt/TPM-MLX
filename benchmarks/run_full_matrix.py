# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import time
import json
import gc
import mlx.core as mx
from pathlib import Path
from tpm_mlx.engine import MLXEngine

MODELS = [
    {
        "name": "Qwen 3.8 27B",
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "draft": "mlx-community/Qwen3.8-27B-MTP-4bit",
        "arch": "27B Dense (Hybrid Linear/Attn)",
        "prompt": "Explain the concept of quantum superposition in two concise sentences.",
    },
    {
        "name": "Qwen 3.8 27B",
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "draft": None,
        "arch": "27B Dense (Hybrid Linear/Attn)",
        "prompt": "Explain the concept of quantum superposition in two concise sentences.",
    },
    {
        "name": "Qwen 3.6 27B",
        "model": "mlx-community/Qwen3.6-27B-4bit",
        "draft": None,
        "arch": "27B Dense Baseline",
        "prompt": "Explain the concept of quantum superposition in two concise sentences.",
    },
    {
        "name": "Muse Glimmer 30B",
        "model": "mlx-community/Muse-Glimmer-30B-4bit",
        "draft": None,
        "arch": "30B Dense Baseline",
        "prompt": "Explain the concept of quantum superposition in two concise sentences.",
    },
    {
        "name": "Qwen 2.5 1.5B Instruct",
        "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "draft": None,
        "arch": "1.5B Dense Edge",
        "prompt": "Explain the concept of quantum superposition in two concise sentences.",
    },
    {
        "name": "Gemma 4 E4B",
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "draft": "mlx-community/gemma-4-E4B-it-assistant-bf16",
        "arch": "4B Dense Edge",
        "prompt": "Explain the difference between supervised and unsupervised learning in two sentences.",
    },
    {
        "name": "Gemma 4 E4B",
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "draft": None,
        "arch": "4B Dense Edge",
        "prompt": "Explain the difference between supervised and unsupervised learning in two sentences.",
    },
    {
        "name": "Gemma 4 E2B",
        "model": "mlx-community/gemma-4-e2b-it-4bit",
        "draft": "mlx-community/gemma-4-E2B-it-assistant-bf16",
        "arch": "2B Dense Edge",
        "prompt": "Explain the difference between supervised and unsupervised learning in two sentences.",
    },
    {
        "name": "Gemma 4 E2B",
        "model": "mlx-community/gemma-4-e2b-it-4bit",
        "draft": None,
        "arch": "2B Dense Edge",
        "prompt": "Explain the difference between supervised and unsupervised learning in two sentences.",
    },
    {
        "name": "Gemma 4 26B-A4B",
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "draft": "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
        "arch": "26B MoE (4B Active)",
        "prompt": "Explain how general relativity describes gravity in two sentences.",
    },
    {
        "name": "Gemma 4 26B-A4B",
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "draft": None,
        "arch": "26B MoE (4B Active)",
        "prompt": "Explain how general relativity describes gravity in two sentences.",
    },
    {
        "name": "Gemma 4 12B QAT",
        "model": "mlx-community/gemma-4-12B-it-qat-4bit",
        "draft": None,
        "arch": "12B Dense",
        "prompt": "Explain the difference between supervised and unsupervised learning in two sentences.",
    },
    {
        "name": "Gemma 3 1B IT",
        "model": "mlx-community/gemma-3-1b-it-4bit",
        "draft": None,
        "arch": "1B Dense Edge",
        "prompt": "Explain what an artificial neural network is in one sentence.",
    },
]

def run_matrix():
    print("=" * 80)
    print("TPM-MLX FULL SANITY & BENCHMARK MATRIX")
    print("=" * 80)
    
    results = []
    max_tokens = 64

    for i, item in enumerate(MODELS, 1):
        name = item["name"]
        model_id = item["model"]
        draft_id = item["draft"]
        arch = item["arch"]
        prompt = item["prompt"]
        spec_label = "MTP / Drafter" if draft_id else "Non-Speculative"

        print(f"\n[{i}/{len(MODELS)}] Testing {name} ({spec_label})")
        print(f"  Model: {model_id}")
        if draft_id:
            print(f"  Draft: {draft_id}")

        mx.clear_cache()
        gc.collect()

        try:
            t0 = time.perf_counter()
            engine = MLXEngine(
                model_path_or_id=model_id,
                draft_model_path_or_id=draft_id,
                max_kv_size=2048,
            )
            load_time = time.perf_counter() - t0

            # Warmup
            list(engine.generate_stream("Warmup query", max_tokens=8))

            # Benchmark
            start_time = time.perf_counter()
            tokens_generated = 0
            ttft = 0.0
            generated_text = ""

            for resp in engine.generate_stream(prompt, max_tokens=max_tokens, temperature=0.0):
                generated_text += resp.text
                tokens_generated = resp.generation_tokens
                if tokens_generated == 1:
                    ttft = (time.perf_counter() - start_time) * 1000.0

            total_time = time.perf_counter() - start_time
            gen_time = total_time - (ttft / 1000.0)
            tps = tokens_generated / gen_time if gen_time > 0 else 0.0
            peak_gb = mx.get_peak_memory() / 1e9

            print(f"  Result: {tps:.2f} TPS | TTFT: {ttft:.1f} ms | Backend: {engine.backend.upper()} | Speculation: {engine.speculation_mode.upper()}")
            print(f"  Output Sample: {generated_text.strip()[:100]}...")

            results.append({
                "name": name,
                "model_id": model_id,
                "draft_id": draft_id,
                "arch": arch,
                "backend": engine.backend.upper(),
                "speculation": engine.speculation_mode.upper(),
                "tps": round(tps, 2),
                "ttft_ms": round(ttft, 1),
                "load_s": round(load_time, 2),
                "tokens": tokens_generated,
                "peak_gb": round(peak_gb, 2),
                "sample": generated_text.strip()[:120],
                "status": "PASS",
            })

            del engine
            mx.clear_cache()
            gc.collect()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "name": name,
                "model_id": model_id,
                "draft_id": draft_id,
                "arch": arch,
                "backend": "ERROR",
                "speculation": "ERROR",
                "tps": 0.0,
                "ttft_ms": 0.0,
                "load_s": 0.0,
                "tokens": 0,
                "peak_gb": 0.0,
                "sample": str(e),
                "status": "FAIL",
            })

    print("\n" + "=" * 80)
    print("BENCHMARK MATRIX RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<22} | {'Speculation':<15} | {'Throughput':<12} | {'TTFT':<10} | {'Backend':<8} | {'Status':<6}")
    print("-" * 80)
    for r in results:
        spec_text = r["speculation"] if r["draft_id"] else "BASELINE"
        print(f"{r['name']:<22} | {spec_text:<15} | {r['tps']:>6.2f} TPS   | {r['ttft_ms']:>6.1f} ms | {r['backend']:<8} | {r['status']:<6}")
    print("=" * 80)

    # Save to JSON
    out_file = Path("/Users/pank/Experiments/MLX/benchmarks/matrix_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {out_file}")

if __name__ == "__main__":
    run_matrix()
