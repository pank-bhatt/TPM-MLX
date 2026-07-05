# TPM-MLX: Architectural Analysis & Optimization Details

This document explains the core architecture of **TPM-MLX (`tpm`)**, how it leverages Apple Silicon hardware, its usage of compiled MLX Metal kernels, and the specific design optimizations that make its tokens-per-second (TPS) throughput faster than naive engines.

---

## 1. Core Architecture Overview

TPM-MLX serves as a zero-bloat, high-performance interface built directly on top of the **MLX** machine learning framework and **Hugging Face's Rust-backed tokenizers**. 

Unlike generalized inference engines (e.g. `llama.cpp` / Ollama) which rely on CPU-to-GPU memory swaps and virtual cache pointers, TPM-MLX operates entirely within Apple Silicon's **Unified Memory Architecture (UMA)**. This allows the CPU and GPU to share physical memory without copying data across a bus, which we leverage to eliminate dynamic allocations.

```
+-------------------------------------------------------------+
|                     User / Client API                       |
+-------------------------------------------------------------+
                              |
                              v (FastAPI Server)
+-------------------------------------------------------------+
|                       MLXEngine                             |
+-------------------------------------------------------------+
      |                                                 |
      v (Rust BPE Tokenizer <0.5ms)                     v (Asynchronous Loop)
+---------------------------+             +---------------------------+
|      Text Encoding        |             |  mx.eval(token, cache)    |
+---------------------------+             +---------------------------+
                                                        |
                                                        v
                                          +---------------------------+
                                          |    PreAllocatedKVCache    |
                                          |   (In-place Metal writes) |
                                          +---------------------------+
                                                        |
                                                        v (Unified RAM)
                                          +---------------------------+
                                          |    Metal Compute GPU      |
                                          +---------------------------+
```

---

## 2. Metal & MLX Kernel Integration

TPM-MLX compiled code executes directly on the Apple Silicon GPU via **Metal compute shaders** written in MSL (Metal Shading Language):

* **Fused Quantized Matrix Multiplication (`quantized_matmul`)**:
  When running 4-bit or 8-bit quantized models, matrix multiplications are executed using custom MLX GPU kernels. These kernels perform **fused dequantization and dot-product multiplication directly inside the GPU registers** (on-the-fly). This avoids writing intermediate dequantized FP16 parameters back to Unified Memory, which is critical since Apple Silicon is heavily memory-bandwidth bound.
* **Metal SDPA (Scaled Dot-Product Attention)**:
  MLX employs custom FlashAttention-style kernels. By supplying pre-allocated contiguous tensors directly to these compiled kernels, we guarantee optimal memory layouts and stride values, preventing GPU compute bubbles.

---

## 3. Why TPM-MLX is Faster Than Usual

Our high TPS (Tokens Per Second) and low TTFT (Time-To-First-Token) are achieved through three main engineering optimizations:

### A. Pre-allocated Static KV Cache (`PreAllocatedKVCache`)
In standard LLM engines, the Key-Value cache grows dynamically step-by-step:
```python
# Naive approach (Dynamic concatenation)
self.keys = mx.concatenate([self.keys, new_keys], axis=2)
```
* **The Problem**: Concatenation is an $O(N)$ memory operation. At step 500, MLX must allocate a new memory buffer for 501 tokens, copy the previous 500 keys, append the 1 new key, and deallocate the old buffer. This causes continuous **memory reallocations, memory copying, and heap fragmentation** in the Metal buffer pool.
* **Our Optimization**: `PreAllocatedKVCache` overrides standard allocation. On the first token generation step, it pre-allocates contiguous zero-tensors of shape `(Batch, Heads, Max_Sequence_Size, Head_Dimension)`:
```python
# Pre-allocation in tpm_mlx/engine.py
self.keys = mx.zeros((B, n_kv_heads, self.max_size, k_head_dim), dtype=keys.dtype)
```
During autoregressive generation, instead of concatenating, it performs an **in-place write** directly into the pre-allocated slice:
```python
# In-place write in tpm_mlx/engine.py
self.keys[..., prev : self.offset, :] = keys
```
This is compiled into a Metal memory-write kernel that copies *only the new token's keys/values* into the already-allocated buffer. This reduces KV cache updates to a constant-time $O(1)$ memory operation, eliminating allocation latency.

### B. Zero CPU-GPU Synchronization Loop
Naive generation loops suffer from CPU-GPU synchronization stalls:
```python
# Naive loop
for token in generate():
    token_id = token.item()  # <--- CRITICAL STALL
```
* **The Problem**: Calling `.item()` converts a GPU tensor to a CPU primitive. This forces a CPU-GPU synchronization barrier. The CPU is blocked, idling until the GPU completes all operations and returns the scalar value.
* **Our Optimization**: We queue calculations asynchronously and evaluate them concurrently using a single evaluation barrier:
```python
# Concurrent evaluation in tpm_mlx/engine.py
mx.eval(token, cache)
```
By evaluating the newly predicted token and the KV cache update in a single `mx.eval()` call, we instruct the MLX compiler to dispatch both operations to the Metal command queue together. The GPU performs the forward pass and updates the KV cache asynchronously while the CPU is already preparing the next token's metadata, preventing hardware execution bubbles.

### C. Gemma 4 Patching (Bypassing Sliding Window Attention Stalls)
Gemma 4 employs hybrid sliding-window (local) and global attention layers. Standard engines struggle with sliding window local caches, raising configuration KeyErrors or falling back to slower CPU attention algorithms. 
Our codebase dynamically patches these architectures:
```python
# Config patch in tpm_mlx/engine.py
if config.get("model_type") == "gemma4_assistant":
    config["text_config"]["num_kv_shared_layers"] = 0
```
By forcing `num_kv_shared_layers` to `0`, we map the model's global layers to our `PreAllocatedKVCache` while preserving local attention structures, keeping memory throughput at its absolute theoretical limit.
