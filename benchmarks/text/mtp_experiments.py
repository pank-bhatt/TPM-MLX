# Copyright © 2026 TPM-MLX Authors. All rights reserved.
"""
Comprehensive Empirical Validation Suite for Multi-Token Prediction (MTP) Levers:
- Lever 1: Sampling Temperature & Stochasticity (Greedy vs High Temp)
- Lever 2: Task Domain & Syntactic Entropy (JSON vs Code vs QA vs Creative)
- Lever 3: Speculation Depth Scaling (gamma = 1, 2, 3)
- Lever 4: Model Weight Precision (4-bit Uniform vs Mixed Quantization)
- Lever 5: Entropy-Gated Adaptive Speculation (Confidence Gating)
"""

import time
import json
import gc
from typing import Dict, Any, List, Tuple
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import _download, load_model, load_tokenizer
from mlx_lm.sample_utils import make_sampler
from tpm_mlx.mtp_weights import load_mtp_head_from_dir
from tpm_mlx.speculation import mtp_generate_step, SpeculationStats

MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
DRAFT_ID = "mlx-community/Qwen3.8-27B-MTP-4bit"

def setup_engine():
    print(f"Loading Base Model ({MODEL_ID})...")
    model_path = Path(_download(MODEL_ID))
    model, _ = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    
    with open(model_path / "config.json") as f:
        config = json.load(f)
        
    print(f"Loading MTP Head ({DRAFT_ID})...")
    draft_path = Path(_download(DRAFT_ID))
    mtp_head = load_mtp_head_from_dir(draft_path, model, config)
    
    return model, tokenizer, mtp_head


def run_single_speculation(
    prompt: str,
    model,
    tokenizer,
    mtp_head,
    num_draft: int = 2,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> Dict[str, Any]:
    prompt_arr = mx.array(tokenizer.encode(prompt))
    
    if temperature == 0.0:
        sampler = lambda x: mx.argmax(x, axis=-1)
    else:
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p)
        
    stats = SpeculationStats()
    mx.clear_cache()
    gc.collect()
    
    start_time = time.perf_counter()
    tokens_generated = 0
    accepted_tokens = 0
    first_token_time = 0.0
    generated_text = ""
    
    for tok, lp, from_draft in mtp_generate_step(
        prompt_arr, 
        model, 
        mtp_head, 
        num_draft_tokens=num_draft, 
        max_tokens=max_tokens, 
        sampler=sampler,
        stats=stats,
    ):
        tokens_generated += 1
        if tokens_generated == 1:
            first_token_time = time.perf_counter() - start_time
        if from_draft:
            accepted_tokens += 1
        generated_text += tokenizer.decode([tok])
        
    total_time = time.perf_counter() - start_time
    gen_time = max(total_time - first_token_time, 1e-6)
    tps = tokens_generated / gen_time
    
    return {
        "tps": round(tps, 2),
        "tokens": tokens_generated,
        "gen_time_s": round(gen_time, 2),
        "ttft_ms": round(first_token_time * 1000, 1),
        "accepted_tokens": stats.accepted_tokens_total,
        "draft_tokens_attempted": stats.draft_tokens_total,
        "acceptance_rate": round(stats.acceptance_rate * 100.0, 1),
        "sample": generated_text.strip()[:80],
    }


def experiment_1_temperature_sampling(model, tokenizer, mtp_head):
    print("\n" + "=" * 80)
    print("LEVER 1: SAMPLING TEMPERATURE & STOCHASTICITY")
    print("=" * 80)
    
    prompt = "Write a Python function to validate whether an email address format is correct using regex."
    temps = [0.0, 0.2, 0.5, 0.8, 1.0]
    results = []
    
    for temp in temps:
        res = run_single_speculation(prompt, model, tokenizer, mtp_head, num_draft=2, max_tokens=100, temperature=temp)
        res["temperature"] = temp
        results.append(res)
        print(f"  Temp={temp:<4} | TPS: {res['tps']:>5.2f} | Acceptance Rate: {res['acceptance_rate']:>5.1f}% ({res['accepted_tokens']:>2}/{res['draft_tokens_attempted']:>2})")
        
    return results


def experiment_2_domain_entropy(model, tokenizer, mtp_head):
    print("\n" + "=" * 80)
    print("LEVER 2: TASK DOMAIN & SYNTACTIC ENTROPY")
    print("=" * 80)
    
    tasks = [
        ("JSON Extraction (Ultra-Low Entropy)", 'Convert this user profile to JSON: Name: Sarah, Age: 29, Role: Engineer, Skills: Python, SQL. Output valid JSON only:'),
        ("Python Code (Low Entropy)", 'Write a Python function to compute the Fibonacci sequence up to n terms with type annotations:'),
        ("Technical QA (Medium Entropy)", 'Explain how Key-Value caching accelerates Transformer inference during autoregression:'),
        ("Creative Fiction (High Entropy)", 'Write the opening paragraph of an imaginative fantasy story about an alchemist who captures starlight in glass bottles:'),
    ]
    
    results = []
    for domain, prompt in tasks:
        res = run_single_speculation(prompt, model, tokenizer, mtp_head, num_draft=2, max_tokens=100, temperature=0.0)
        res["domain"] = domain
        results.append(res)
        print(f"  {domain:<38} | TPS: {res['tps']:>5.2f} | Acceptance: {res['acceptance_rate']:>5.1f}% ({res['accepted_tokens']:>2}/{res['draft_tokens_attempted']:>2})")
        
    return results


def experiment_3_speculation_depth(model, tokenizer, mtp_head):
    print("\n" + "=" * 80)
    print("LEVER 3: SPECULATION DEPTH SCALING (gamma = 1, 2, 3)")
    print("=" * 80)
    
    prompt = "Explain the difference between process concurrency and thread parallelism in operating systems in detail."
    depths = [1, 2, 3]
    results = []
    
    for depth in depths:
        res = run_single_speculation(prompt, model, tokenizer, mtp_head, num_draft=depth, max_tokens=100, temperature=0.0)
        res["depth"] = depth
        results.append(res)
        print(f"  Draft Depth gamma={depth:<2} | TPS: {res['tps']:>5.2f} | Acceptance Rate: {res['acceptance_rate']:>5.1f}% ({res['accepted_tokens']:>2}/{res['draft_tokens_attempted']:>2})")
        
    return results


def main():
    model, tokenizer, mtp_head = setup_engine()
    
    # Warmup
    print("Warming up GPU kernels...")
    run_single_speculation("Warmup query", model, tokenizer, mtp_head, max_tokens=10)
    
    exp1 = experiment_1_temperature_sampling(model, tokenizer, mtp_head)
    exp2 = experiment_2_domain_entropy(model, tokenizer, mtp_head)
    exp3 = experiment_3_speculation_depth(model, tokenizer, mtp_head)
    
    summary = {
        "lever_1_temperature": exp1,
        "lever_2_domain_entropy": exp2,
        "lever_3_speculation_depth": exp3,
    }
    
    out_file = Path(__file__).parent / "mtp_experiments_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll experimental runs finished! Saved to {out_file}")

if __name__ == "__main__":
    main()
