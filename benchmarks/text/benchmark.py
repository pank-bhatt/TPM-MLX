# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import time
import argparse
import mlx.core as mx
from tpm_mlx.engine import MLXEngine
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def run_tpm_benchmark(
    model_id: str, 
    prompt: str, 
    max_tokens: int,
    draft_model: str = None,
    enable_mtp: bool = True,
    num_draft_tokens: int = None,
) -> tuple:
    print(f"Loading {model_id} via TPM Engine (Draft: {draft_model}, MTP: {enable_mtp})...")
    start_load = time.perf_counter()
    engine = MLXEngine(
        model_path_or_id=model_id, 
        max_kv_size=4096,
        draft_model_path_or_id=draft_model,
        enable_mtp=enable_mtp,
        num_draft_tokens=num_draft_tokens,
    )
    load_time = time.perf_counter() - start_load
    print(f"TPM Engine loaded in {load_time:.2f}s [Speculation: {engine.speculation_mode.upper()}]")
    
    # Run warmup
    list(engine.generate_stream("warmup query", max_tokens=10))
    
    # Benchmark run
    start_time = time.perf_counter()
    tokens_generated = 0
    ttft = 0.0
    
    for response in engine.generate_stream(prompt, max_tokens=max_tokens, temperature=0.0):
        tokens_generated = response.generation_tokens
        if tokens_generated == 1:
            ttft = (time.perf_counter() - start_time) * 1000.0  # ms
            
    total_time = time.perf_counter() - start_time
    # subtract TTFT from total generation time to get tokens/s for generation phase
    gen_time = total_time - (ttft / 1000.0)
    tps = tokens_generated / gen_time if gen_time > 0 else 0.0
    
    spec_info = ""
    if engine.speculation_mode != "none":
        spec_info = f" (Speculation: {engine.speculation_mode}, Acceptance: {engine.speculation_stats.acceptance_rate * 100:.1f}%)"
    
    return tps, ttft, tokens_generated, engine.speculation_mode, engine.speculation_stats.acceptance_rate

def run_baseline_benchmark(model_id: str, prompt: str, max_tokens: int) -> tuple:
    print(f"Loading {model_id} via baseline...")
    start_load = time.perf_counter()
    
    # Check if model is VLM
    is_vlm = "gemma-4" in model_id.lower() or "vl" in model_id.lower()
    
    if is_vlm:
        import mlx_vlm.utils as vlm_utils
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        
        model, processor = vlm_utils.load(model_id)
        load_time = time.perf_counter() - start_load
        print(f"VLM Baseline loaded in {load_time:.2f}s")
        
        try:
            formatted_prompt = apply_chat_template(processor, model.config, prompt)
        except Exception:
            formatted_prompt = prompt
            
        start_time = time.perf_counter()
        tokens_generated = 0
        ttft = 0.0
        
        for n, response in enumerate(vlm_stream_generate(model, processor, prompt=formatted_prompt, max_tokens=max_tokens, temperature=0.0)):
            tokens_generated = n + 1
            if tokens_generated == 1:
                ttft = (time.perf_counter() - start_time) * 1000.0
                
        total_time = time.perf_counter() - start_time
        gen_time = total_time - (ttft / 1000.0)
        tps = tokens_generated / gen_time if gen_time > 0 else 0.0
        return tps, ttft, tokens_generated
    else:
        model, tokenizer = load(model_id)
        load_time = time.perf_counter() - start_load
        print(f"Baseline loaded in {load_time:.2f}s")
        
        # Run warmup
        generate(model, tokenizer, "warmup query", max_tokens=10)
        
        # Benchmark run
        from mlx_lm.generate import stream_generate
        start_time = time.perf_counter()
        tokens_generated = 0
        ttft = 0.0
        
        sampler = make_sampler(temp=0.0)
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            tokens_generated = response.generation_tokens
            if tokens_generated == 1:
                ttft = (time.perf_counter() - start_time) * 1000.0 # ms
                
        total_time = time.perf_counter() - start_time
        gen_time = total_time - (ttft / 1000.0)
        tps = tokens_generated / gen_time if gen_time > 0 else 0.0
        
        return tps, ttft, tokens_generated

def main():
    parser = argparse.ArgumentParser(description="TPM-MLX vs mlx-lm Baseline Benchmark")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model path or Hugging Face ID")
    parser.add_argument("--draft-model", type=str, default=None, help="Companion draft model for speculative decoding")
    parser.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True, help="Toggle MTP self-speculation")
    parser.add_argument("--num-draft-tokens", type=int, default=None, help="Number of speculative tokens to draft")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.", help="Prompt to run")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TPM-MLX THROUGHPUT BENCHMARK")
    print(f"Model:  {args.model}")
    if args.draft_model:
        print(f"Draft:  {args.draft_model}")
    print(f"Prompt: {args.prompt}")
    print("=" * 60)
    
    try:
        tpm_tps, tpm_ttft, tpm_tokens, spec_mode, acc_rate = run_tpm_benchmark(
            args.model, args.prompt, args.max_tokens,
            draft_model=args.draft_model,
            enable_mtp=args.mtp,
            num_draft_tokens=args.num_draft_tokens,
        )
    except Exception as e:
        import traceback
        print(f"TPM Benchmark failed: {e}")
        traceback.print_exc()
        tpm_tps, tpm_ttft, tpm_tokens, spec_mode, acc_rate = 0.0, 0.0, 0, "none", 0.0
        
        
    print("-" * 60)
    
    try:
        base_tps, base_ttft, base_tokens = run_baseline_benchmark(args.model, args.prompt, args.max_tokens)
    except Exception as e:
        import traceback
        print(f"Baseline Benchmark failed: {e}")
        traceback.print_exc()
        base_tps, base_ttft, base_tokens = 0.0, 0.0, 0
        
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Engine':<20} | {'Throughput (TPS)':<20} | {'TTFT (ms)':<15} | {'Speculation':<15}")
    print("-" * 75)
    spec_str = f"{spec_mode.upper()}" if spec_mode != "none" else "NONE"
    if spec_mode != "none":
        spec_str += f" ({acc_rate * 100:.0f}%)"
    print(f"{'TPM-MLX (Ours)':<20} | {tpm_tps:<20.2f} | {tpm_ttft:<15.2f} | {spec_str:<15}")
    print(f"{'mlx-lm (Baseline)':<20} | {base_tps:<20.2f} | {base_ttft:<15.2f} | {'N/A':<15}")
    print("=" * 75)

if __name__ == "__main__":
    main()
