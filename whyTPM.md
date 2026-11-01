# TPM-MLX: Architectural Analysis & Optimization Details

This document provides a comprehensive technical breakdown of **TPM-MLX (`tpm`)**, explaining how it maximizes throughput (`#ThroughPutMaxing`) across **LLMs, Vision-Language Models, Flow Matching Image Diffusion, and 3D Mesh Synthesis** on Apple Silicon hardware through unified memory design, compiled Metal kernels, speculative decoding, and strict lifecycle orchestration.

---

## 1. Unified Architecture Overview

Unlike traditional ML inference engines (e.g. `llama.cpp`, Ollama, vLLM) that are architected around discrete CPU-GPU setups with PCIe bus memory transfers and virtual cache pointers, **TPM-MLX operates natively on Apple Silicon's Unified Memory Architecture (UMA)**. 

CPU cores, Neural Engine, and Metal GPU execution units access a single physical, high-bandwidth memory pool (up to 800+ GB/s on M-series Max/Ultra chips). TPM-MLX eliminates host-to-device memory copies, dynamic heap reallocation, and CPU-GPU synchronization stalls across all supported modalities.

```
+-----------------------------------------------------------------------------------+
|                        Client Layer (OpenAI API / Web Playground)                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v (FastAPI & Dynamic Dispatcher)
+-----------------------------------------------------------------------------------+
|                                TPM-MLX State & Concurrency Guard                  |
|                 active_inference() locks  |  Safe Idle Eviction Watchdog          |
+-----------------------------------------------------------------------------------+
          |                               |                                |
          v                               v                                v
+--------------------+          +--------------------+          +--------------------+
|   LLM / VLM Engine |          | Image Engine (DiT) |          |  3D Engine (DiT)   |
|   mlx-lm / mlx-vlm |          | FLUX.2 / Z-Image   |          | Hunyuan3D 2.1 MLX  |
|   MTP / Gemma Drafter         | Qwen-Image / Bonsai|          | ShapeVAE + PBR/Bake|
+--------------------+          +--------------------+          +--------------------+
          |                               |                                |
          +-------------------------------+--------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Unified Apple Silicon Metal Subsystem                      |
|  PreAllocatedKVCache  |  Fused Quant Matmul  |  Metal FWHT  |  Phase Cache Flush |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                   Unified Memory Architecture (UMA Physical RAM Pool)             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Metal & MLX Kernel Integration

TPM-MLX compiled code executes directly on Apple Silicon GPU cores via **Metal compute shaders** written in MSL (Metal Shading Language):

### A. Fused Quantized Matrix Multiplication (`quantized_matmul`)
When evaluating 4-bit, 8-bit, or ternary quantized weights, matrix multiplications are executed using custom MLX Metal kernels. These kernels perform **fused dequantization and dot-product multiplication directly inside the GPU registers on-the-fly**. 
* **The Benefit:** Avoids decompressing FP16 parameters into memory buffers. Because Apple Silicon is primarily memory-bandwidth bound during autoregressive generation, keeping weights compact in memory and dequantizing in registers directly scales token generation throughput.

### B. Hardware-Accelerated Fast Walsh-Hadamard Transform (FWHT)
For ultra-low-bit models like **Ternary Bonsai 2 27B** (`prism_hadamard_qwen35`):
* TPM-MLX utilizes compiled Metal Fast Walsh-Hadamard Transform kernels (`fwht`) and 2-bit packed layers.
* This unlocks full 27-billion parameter multimodal reasoning and vision in **~8.6 GB of Unified Memory** at over **20 TPS**, making high-capability AI practical on consumer 16 GB and 24 GB Macs.

### C. Metal SDPA (Scaled Dot-Product Attention)
MLX employs FlashAttention-style Metal kernels. Supplying contiguous pre-allocated memory slices ensures optimal memory layout and stride alignments, preventing GPU execution stalls.

---

## 3. Core LLM Optimizations

### A. Pre-allocated Static KV Cache (`PreAllocatedKVCache`)
In standard inference engines, the Key-Value cache grows dynamically step-by-step:
```python
# Naive approach (Dynamic reallocation)
self.keys = mx.concatenate([self.keys, new_keys], axis=2)
```
* **The Problem:** Concatenation is an $O(N)$ memory operation. At token 1000, MLX must allocate a new buffer for 1001 tokens, copy the previous 1000 keys, append the new key, and release the old buffer. This causes continuous **Metal heap fragmentation, reallocation overhead, and latency spikes**.
* **TPM-MLX Solution:** `PreAllocatedKVCache` pre-allocates contiguous memory slices up to the configured limit (e.g. `--max-kv-size 32768`) during the initial forward pass:
```python
# Pre-allocation in tpm_mlx/cache.py
self.keys = mx.zeros((B, n_kv_heads, self.max_size, k_head_dim), dtype=keys.dtype)
```
During autoregressive generation, new keys/values are written directly in-place:
```python
# In-place write: constant-time O(1)
self.keys[..., prev : self.offset, :] = keys
```
This reduces cache updates to constant-time $O(1)$ in-place Metal writes, eliminating memory reallocation latency.

### B. Zero CPU-GPU Synchronization Loop
Naive loops synchronize CPU and GPU on every token:
```python
# Naive stall
for token in generate():
    token_id = token.item()  # <--- Blocks CPU until GPU completes
```
* **The Problem:** Calling `.item()` on a GPU tensor forces a synchronous hardware barrier. The CPU sits idle waiting for the Metal command queue to finish.
* **TPM-MLX Solution:** We queue computation asynchronously and trigger execution through a single evaluation barrier:
```python
# Single evaluation barrier in tpm_mlx/engine.py
mx.eval(token, cache)
```
Both the predicted token and the in-place KV cache updates are dispatched to the Metal command queue in a single batch. The GPU executes the forward pass while the CPU concurrently prepares the subsequent step metadata.

---

## 4. Native Multi-Token Prediction (MTP) Self-Speculation

TPM-MLX implements native speculative decoding for models with MTP heads (e.g. **Qwen 3.x, Xiaomi MiMo, DeepSeek-V3**):

```
Step t:
  Backbone Forward Pass --------> Output Hidden State h_t ----> Main Token t
                                           |
                                           v
                                   MTP Head Forward Pass ------> Draft Token t+1 (Zero Latency)
```

1. **Trunk Feature Reuse:** Standard speculative decoding requires running a separate draft model (which incurs additional memory bandwidth overhead). In contrast, MTP extracts speculative tokens directly from the main transformer trunk's final hidden states with negligible compute overhead.
2. **Recurrent Cache Rollback without Memory Copies:** For hybrid linear/attention architectures (like Qwen 3.8), TPM-MLX implements stateful recurrent cache snapshotting (`_snapshot_recurrent`) and prefix token replay on `ArraysCache`. If a speculative token is rejected, the cache state is rewound in constant time without memory reallocations.
3. **Throughput Impact:** Delivers up to a **1.61× (61%) throughput improvement** on 27B models (reaching 25.5 TPS on Apple Silicon) with a **62% average draft acceptance rate**.

---

## 5. Unified Memory Diffusion & 3D Synthesis

Running massive Diffusion Transformer (DiT) models (such as **FLUX.2 Klein 9B**, **Qwen-Image 2.1**, or **Hunyuan3D 2.1**) on discrete GPUs typically requires balancing tight VRAM constraints with PCIe transfer penalties. On Apple Silicon, UMA allows DiT models to access the entire unified memory pool directly.

However, running Text + 2D Diffusion + 3D Mesh Extraction concurrently can easily spike beyond 50–60 GB if unmanaged. TPM-MLX introduces two core memory controls:

### A. Strict Sequential Chaining
For end-to-end `Text → 2D → 3D` flows:
1. The 3D engine unloads resident 3D pipeline weights before triggering the 2D diffusion engine (e.g. Qwen-Image 2.1) to generate the reference image.
2. Upon 2D completion, the image engine weights and intermediate activations are completely evicted from Unified Memory before loading the 3D Shape DiT and DINOv2 encoders.
3. This guarantees that peak activations from 2D diffusion (~22 GB) and 3D DiT (~15 GB) never overlap, preventing system memory exhaustion and swap thrashing.

### B. Intermediate Phase Memory Flushing
High-resolution diffusion pipelines generate substantial transient Metal command buffer allocations:
* In `Hunyuan3D 2.1`, TPM-MLX invokes `mx.clear_cache()` periodically during iterative DiT steps and explicitly unloads conditioning tensors before executing Shape VAE marching cubes and CPU mesh decimation.
* Provides **Fast Preview (Direct UV Bake)** mode: projects and bakes reference textures directly into geometry UV space in **3–5 seconds** with **0 extra VRAM**, offering an ultra-fast alternative to 10-minute multi-view diffusion painting.

---

## 6. Zero-Downtime Lifecycle & Concurrency Guarantees

TPM-MLX ensures reliable long-running server operation through multi-engine lifecycle management:

```
                  +-----------------------------------+
                  |   Incoming Request (Chat/3D/Img)  |
                  +-----------------------------------+
                                    |
                                    v
                     +------------------------------+
                     | active_inference(modality)   |
                     | Increments busy_counts[mod]  |
                     +------------------------------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v (During Computation)                v (Background Watchdog)
      [Execute Metal Inference]               [Inspect Idle Models]
                 |                            Is busy_counts == 0?
                 v                            Is (now - last_completed) > timeout?
     [Decrement busy_counts]                           |
     [Update last_completed]                 +---------+---------+
                 |                           |                   |
                 v                         (Yes)                (No)
         [Return Response]                   v                   v
                                      [Safe Eviction]      [Retain in RAM]
```

1. **Active Inference Guards:** All inference operations are wrapped in `@contextmanager active_inference(modality)`. The background idle watchdog and manual ejection routes verify that `busy_counts == 0` and no async tasks are queued before initiating model eviction.
2. **Safe Idle Eviction Watchdog:** Background thread wakes every 10 seconds to check model inactivity. Unused engines are safely evicted after their configured timeout (e.g. 5m for Image/3D, 15m for Text).
3. **One-Click Reclaim:** `POST /v1/unload` allows users or client applications to immediately drop standing memory to 0 MB. Models reload on-demand upon the next request.
