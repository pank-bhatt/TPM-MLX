# TPM-MLX 3D Generation Integration Plan (`Text / Image → 3D`)

**Status:** Approved Specification  
**Target Repository:** [`pank-bhatt/TPM-MLX`](https://github.com/pank-bhatt/TPM-MLX)  
**Primary Engine:** Tencent **Hunyuan3D 2.1** (`0.6B-mini` & `1.1B-turbo` Shape DiT + Multi-View PBR Paint UNet)  
**Secondary Engine (Phase 4):** Microsoft **TRELLIS.2-4B** (`O-Voxel` Sparse 3D VAE + DiT for open/non-manifold surfaces)

---

## 1. Executive Summary & Architectural Fit

**TPM-MLX** is built around `#ThroughPutMaxing` on Apple Silicon—combining zero-sync execution loops, low-bit quantization, speculative LLM decoding, pre-allocated KV caches, and native flow-matching 2D diffusion (**FLUX.2**, **Z-Image Turbo**, **Qwen-Image 2.1**, **Bonsai**) inside a local, zero-telemetry studio with OpenAI-compatible APIs and a glassmorphic Web Playground on port `2505`.

Integrating **Hunyuan3D 2.1** extends TPM-MLX into a complete **Text → 2D → 3D** and **Image → 3D** multimodal sovereign pipeline while reusing the core architectural principles of the project:

1. **Shared Mathematical Backbone (Flow Matching):**
   Like FLUX.2, Z-Image Turbo, and Qwen-Image 2.1, `Hunyuan3D-Shape` is a **Flow-Matching Diffusion Transformer (DiT)**. Its `1.1B-turbo` variant uses **8-step consistency distillation**, aligning with the low-step, high-throughput execution path already established in `TPM-MLX`.
2. **Zero-Copy In-Memory Chaining (`Text → 2D → 3D`):**
   Users can synthesize a 2D reference image with FLUX.2 or Qwen-Image 2.1 and pipe the resulting `mx.array` tensor directly into the 3D conditioning pipeline without disk serialization or CPU-GPU round-trips.
3. **100% Native Apple Silicon Execution (Zero CUDA / PyTorch Bridge Overhead):**
   By combining native MLX ports with tiled Metal UV rasterization and `xatlas`, the entire pipeline runs natively on Apple Silicon without foreign CUDA C++ extensions (`nvdiffrast`, `spconv`, `cumesh`), achieving peak memory efficiency on unified memory.
4. **Decoupled 2-Stage Pipeline (Instant Mesh vs. 4K PBR):**
   Users get an interactive, watertight geometry `.glb` in **~15–20s** (`~5–7 GB` VRAM) and can selectively trigger Stage 2 multi-view PBR texture synthesis (`~11–14 GB` VRAM).

---

## 2. End-to-End Pipeline & Memory Telemetry

The 3D subsystem decouples **Stage 1 (Geometry Generation)** from **Stage 2 (Multi-View PBR Texture Synthesis)**:

| Stage | Sub-Module | Input → Output Tensor Shapes | FP16 Peak VRAM | INT8 Peak VRAM | INT4 Peak VRAM | Target Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **0. Pre-Processing** | `BiRefNet` / `rembg` (MLX) | `RGB Image` → `518×518 RGBA` (white-composited) | ~1.2 GB | ~0.6 GB | ~0.6 GB | **~0.4s** |
| **1A. Vision Encoder** | `DINOv2-giant` (`518×518` ViT) | `(1, 3, 518, 518)` → `(1, 1370, 1536)` patch + CLS tokens | ~2.3 GB | ~1.1 GB | ~1.1 GB | **~0.6s** |
| **1B. Shape Diffusion** | `Hunyuan3D-DiT` (`0.6B` / `1.1B-turbo`) | Noise `(1, 3072, 64)` + DINOv2 → Latents `(1, 3072, 64)` | ~5.7–7.3 GB | ~3.0–3.8 GB | ~1.6–2.2 GB | **~12–16s** (8 steps) |
| **1C. Mesh Extraction** | `ShapeVAE` + Octree SDF + Marching Cubes | `(1, 3072, 64)` → `256³` SDF Grid → Watertight `.glb` | ~2.0 GB | ~1.5 GB | ~1.5 GB | **~2–4s** |
| **2A. Remesh & UV** | Mesh Decimation + `xatlas` + Metal Normal/Pos Render | Raw `~500K` faces → `~40K` faces + 6 views (`512×512`) | ~1.5 GB | ~1.5 GB | ~1.5 GB | **~3–5s** |
| **2B. Multi-View PBR** | `Hunyuan3D-Paint` (2.5D UNet + VAE + RealESRGAN 4×) | 6 Views + Ref Image → 6× `2048²` Albedo/MR → `4096²` PBR `.glb` | ~25.0–35.0 GB | ~11.0–14.0 GB | ~8.5–11.0 GB | **~45–70s** |

---

## 3. Repository Architecture & File Mapping

The 3D implementation strictly adheres to TPM-MLX's decoupled, layered architecture:

```text
TPM-MLX/
├── src/
│   └── tpm_mlx/
│       ├── engine_3d.py             # Core MLX3DEngine (lazy loader, inference orchestrator, export)
│       ├── state.py                 # Centralized state + unload_3d_engine() + mutual eviction
│       ├── routes/
│       │   ├── __init__.py          # Export router_3d
│       │   ├── threed.py            # POST /v1/3d/generations & SSE telemetry stream
│       │   └── models.py            # 3D models registry in /v1/models (KNOWN_3D_MODELS)
│       ├── schemas/
│       │   ├── __init__.py          # Export 3D request/response schemas
│       │   └── threed.py            # Pydantic v2 validation (ThreeDGenerationRequest, ThreeDMetrics)
│       ├── diffusion_3d/            # Internal 3D mathematical kernels and pipelines
│       │   ├── __init__.py
│       │   ├── config.py            # Model parameters, octree thresholds, scheduler sigmas
│       │   ├── preprocessor.py      # MLX Alpha matting (BiRefNet) & 518x518 white-matte centering
│       │   ├── dinov2_mlx.py        # Native MLX DINOv2-giant ViT feature extractor
│       │   ├── shape_dit.py         # 21-block Flow-Matching DiT (standard MLPs + MoE blocks)
│       │   ├── shape_vae.py         # Vectset ShapeVAE decoder + coarse-to-fine octree SDF evaluator
│       │   ├── mesh_extract.py      # Fast Marching Cubes, quadric decimation, GLB/USDZ export
│       │   ├── paint_unet.py        # 2.5D multi-view UNet with reference cross-attention
│       │   ├── texture_baker.py     # xatlas UV unwrap, multi-view cosine blend, EDT gutter padding
│       │   └── metal_kernels.py     # Tiled UV rasterizer & custom mx.fast.metal_kernel ops
│       └── static/
│           ├── models_3d/           # Storage for generated .glb & .usdz mesh assets
│           ├── playground.html      # <model-viewer> 3D viewport modal and controls
│           └── js/
│               ├── app.js           # "🧊 Convert to 3D" action, 3D metric cards
│               └── viewer_3d.js     # Three.js / <model-viewer> controller (PBR lit, Wireframe, AR)
├── benchmarks/
│   ├── test_3d_engine.py            # Unit tests (DINOv2, DiT shapes, SDF grid, GLB export)
│   └── image/                       # Consolidated image benchmarks (Phase 1 & Phase 2)
└── pyproject.toml                   # [project.optional-dependencies] 3d = [...]
```

---

## 4. Why This Plan is Optimized for MLX and `#ThroughPutMaxing`

### 4.1 Zero CPU-GPU Synchronization Euler Loop
* **Standard Bottleneck:** Naive implementations evaluate tensor shapes, inspect intermediate values, or update progress bars inside the 8-step denoising loop. Every step forces a synchronization barrier where the CPU idles waiting for the Apple Silicon GPU.
* **TPM-MLX Optimization:**
  * Transformer blocks are decorated with `@mx.compile` for kernel fusion.
  * Reuses TPM-MLX's asynchronous loop: `latents = latents + dt * v_pred`, queued with `mx.async_eval(latents)`.
  * The GPU executes all 8 consistency-distilled steps without a single CPU stall, synchronizing only once before mesh extraction.

### 4.2 Coarse-to-Fine Octree SDF Pruning (>85% Compute Reduction)
* **Standard Bottleneck:** Querying `ShapeVAE` across a uniform $256^3$ volumetric grid requires **16,777,216 cross-attention evaluations**, even though >90% of the volume is empty air.
* **TPM-MLX Optimization:**
  * Hierarchical evaluation: `32³ → 64³ → 128³ → 256³`.
  * At each level, voxels where `mx.abs(sdf) > threshold` are immediately pruned.
  * Surviving surface voxels are queried in contiguous batches of `32,768` coordinates to maximize Apple Silicon Metal SIMD lane saturation.

### 4.3 Tiled Metal UV Rasterization & Mesh Decimation
* **Standard Bottleneck:** High-density Marching Cubes generates `~300K–500K` triangles. Dispatching a single monolithic `4096×4096` UV rasterization pass triggers the macOS Metal watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`), causing kernel panics or process kills.
* **TPM-MLX Optimization:**
  * **Automatic Quadric Decimation:** Pre-simplifies the raw mesh to `~40,000` faces before UV parameterization, preserving sharp geometric silhouettes while cutting UV unwrapping time by 75%.
  * **Tiled Dispatch:** Partitions the `4096×4096` UV canvas into spatial tiles, ensuring every command buffer finishes in `<50 ms` with zero UI frame drops.
  * **Seam-Free EDT Padding:** Cosine-weighted multi-view blending is followed by an Euclidean Distance Transform (EDT) nearest-fill post-pass to pad UV gutters, eliminating dark border seams during bilinear mipmap filtering.

### 4.4 INT4 / INT8 Mixed Quantization
* **Shape DiT:** Blocks 0–14 use standard MLPs, while Blocks 15–20 use an 8-expert Mixture-of-Experts (`top_k=2`) architecture.
* Quantizing linear and MoE layers to `4-bit` (`group_size=64`) via `mx.quantize()`—while keeping `RMSNorm`, timesteps, and cross-attention projections in `float16`—reduces the DiT memory footprint from **5.7 GB to 1.6 GB**, doubling token throughput on memory-bandwidth-bound M-series chips.

### 4.5 Mutual Unified Memory Eviction
Integrated directly into [`src/tpm_mlx/state.py`](file:///Users/pank/Experiments/MLX/src/tpm_mlx/state.py):
* When a 3D model loads, `unload_text_engine()`, `unload_image_engine()`, and `unload_embedding_engine()` are invoked automatically.
* When returning to LLM chat or 2D image diffusion, `unload_3d_engine()` releases the Shape DiT, DINOv2, and Paint UNet buffers and clears Metal GPU caches via `mx.metal.clear_cache()` and `mx.reset_peak_memory()`.

---

## 5. API & Web Playground Specification

### 5.1 Endpoint (`POST /v1/3d/generations`)

#### Request Payload
```json
{
  "model": "hunyuan3d-2.1-turbo-mlx",
  "image": "/images/gen_179180.png",
  "prompt": null,
  "steps": 8,
  "octree_resolution": 256,
  "target_face_count": 40000,
  "pbr_paint": false,
  "texture_resolution": 2048,
  "export_formats": ["glb", "usdz"],
  "response_format": "url"
}
```

#### JSON Response
```json
{
  "created": 1792000000,
  "data": [
    {
      "glb_url": "/models_3d/mesh_1792000000_1234.glb",
      "usdz_url": "/models_3d/mesh_1792000000_1234.usdz"
    }
  ],
  "tpm_metrics": {
    "generation_time_s": 17.84,
    "faces": 39820,
    "vertices": 20114,
    "peak_memory_gb": 6.84,
    "has_pbr_texture": false,
    "octree_resolution": 256
  }
}
```

### 5.2 Interactive 3D Web Studio & AR QuickLook
* **`<model-viewer>` Integration:** Clean, responsive 3D card embedded in the multi-session chat history with Orbit Controls, Auto-Rotate, and Shading toggles (`PBR Lit`, `Clay MatCap`, `Wireframe`).
* **"🧊 Convert to 3D" One-Click Action:** Every 2D image card generated by FLUX.2, Qwen-Image 2.1, or Bonsai displays a `🧊 Convert to 3D` button that submits the image directly to `/v1/3d/generations`.
* **iOS / iPadOS AR QuickLook:** Attaches `.usdz` format dynamically (`ios-src`), allowing mobile devices connected over local Wi-Fi or Tailscale to view and place the 3D asset in physical augmented reality with one tap.

---

## 6. Phased Implementation Roadmap

### Phase 1: Stage 1 Geometry MVP (Watertight Clay Meshes in ~18s)
- [x] Add `[project.optional-dependencies] 3d = [...]` in `pyproject.toml` (`trimesh`, `pygltflib`, `rembg`, `scipy`, `xatlas`, `scikit-image`, `mlx-arsenal`).
- [x] Implement `src/tpm_mlx/diffusion_3d/`:
  - `preprocessor.py` (rembg alpha matting + 518×518 centering & white matte composite).
  - `dinov2_mlx.py` (Native MLX DINOv2 ViT).
  - `shape_dit.py` (Hunyuan3D-DiT 1.1B-turbo with 8-step consistency scheduler and MoE blocks 15-20).
  - `shape_vae.py` (Vectset ShapeVAE with coarse-to-fine octree SDF).
  - `mesh_extract.py` (Marching Cubes + fast decimation + `.glb` / `.usdz` export).
- [x] Wire `engine_3d.py`, `state.py` (`unload_3d_engine`), `routes/threed.py`, and `schemas/threed.py`.
- [x] Add unit test suite `benchmarks/test_3d_engine.py` (100% test pass).
- [x] Add CLI command `tpm generate-3d --image input.png --steps 8 -o output.glb`.

### Phase 2: Web Studio 3D Viewer & 2D-to-3D Chaining
- [x] Update `src/tpm_mlx/static/playground.html` with Google `<model-viewer>` CDN and 3D preview modal.
- [x] Add `🧊 Convert to 3D` button to image cards in `src/tpm_mlx/static/js/app.js` and `lightbox.js`.
- [x] Wire live telemetry display (generation time, face count, peak VRAM).

### Phase 3: Stage 2 Multi-View PBR Texture Synthesis
- [ ] Implement `xatlas` UV parameterization and tiled Metal rasterizer (`metal_kernels.py`).
- [ ] Implement `paint_unet.py` (2.5D multi-view UNet with reference attention) and MLX RealESRGAN x4.
- [ ] Add cosine normal blending, EDT gutter inpainting, and glTF 2.0 PBR material export.

### Phase 4: TRELLIS.2-4B O-Voxel Extension (Advanced Topology)
- [ ] Integrate optional TRELLIS.2 backend for non-manifold structures, thin foliage, and transparent assets.