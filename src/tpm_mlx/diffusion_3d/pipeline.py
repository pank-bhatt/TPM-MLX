"""End-to-end Hunyuan3D-2.1 Shape Generation Pipeline in MLX.

Orchestrates:
1. Input Image Preprocessing (rembg + recentering + white composite)
2. Vision Conditioning (DINOv2 ViT)
3. Zero-Sync Flow-Matching Diffusion (8-step HunYuanDiTPlain)
4. Hierarchical Octree SDF Decoding (ShapeVAEDecoder)
5. Watertight Polygon Reconstruct & Simplification (skimage + trimesh)
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image
import trimesh

from .config import Hunyuan3DConfig
from .dinov2_mlx import ImageEncoder
from .mesh_extract import clean_mesh, export_mesh_files, simplify_mesh
from .preprocessor import ImageProcessorV2
from .shape_dit import HunYuanDiTPlain
from .shape_vae import ShapeVAEDecoder

logger = logging.getLogger(__name__)


class FlowMatchEulerScheduler:
    """Uniform Flow-Matching Euler scheduler for Hunyuan3D-2.1.

    Integrates from t=0 (noise) to t=1 (data) matching Hunyuan3D-2.1 flow matching ODE.
    """

    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps: int = 30
        self.timesteps: list[float] = []
        self.sigmas: np.ndarray = np.array([])
        self._step_index: int = 0

    def set_timesteps(self, num_inference_steps: int, sigmas: Optional[np.ndarray] = None) -> None:
        if sigmas is None:
            sigmas_arr = np.linspace(0, 1, num_inference_steps, dtype=np.float32)
        else:
            sigmas_arr = np.asarray(sigmas, dtype=np.float32)

        self.num_inference_steps = num_inference_steps
        self.timesteps = [float(s * self.num_train_timesteps) for s in sigmas_arr]
        # Append terminal sigma 1.0 (matching Hunyuan3D PyTorch: torch.cat([sigmas, torch.ones(1)]))
        self.sigmas = np.concatenate([sigmas_arr, np.ones(1, dtype=np.float32)])
        self._step_index = 0

    def step(self, model_output: mx.array, sample: mx.array, timestep: Any = None) -> mx.array:
        del timestep
        if self._step_index >= len(self.sigmas) - 1:
            dt = 1.0 / max(1, self.num_inference_steps)
        else:
            dt = float(self.sigmas[self._step_index + 1] - self.sigmas[self._step_index])
        prev_sample = sample + dt * model_output
        self._step_index += 1
        return prev_sample


class Hunyuan3DShapePipeline:
    """Zero-sync Apple Silicon native pipeline for Hunyuan3D shape generation."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        dit: HunYuanDiTPlain,
        vae: ShapeVAEDecoder,
        scheduler: Optional[Any] = None,
        dtype: mx.Dtype = mx.float16,
    ):
        self.image_encoder = image_encoder
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler or FlowMatchEulerScheduler()
        self.preprocessor = ImageProcessorV2(size=518, border_ratio=0.15)
        self.dtype = dtype

        # Ensure all models are in target dtype
        self.image_encoder.set_dtype(dtype)
        self.dit.set_dtype(dtype)
        self.vae.set_dtype(dtype)

    def generate(
        self,
        image: Union[str, Path, Image.Image, mx.array],
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
        target_face_count: int = 40000,
        bounds: float = 1.01,
        mc_level: float = 0.0,
        seed: Optional[int] = None,
        auto_remove_bg: bool = True,
        num_chunks: int = 32768,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """Executes full Image-to-3D shape generation.

        Args:
            image: Input image path, PIL image, or tensor.
            num_inference_steps: Denoising steps (8 for turbo, 30-50 for base).
            guidance_scale: Classifier-free guidance scale (0.0 for turbo distilled).
            octree_resolution: Grid resolution for marching cubes (default 256).
            target_face_count: Target decimated polygon face count (default 40,000).
            bounds: Bounding box scale.
            mc_level: Marching cubes iso-level.
            seed: Optional deterministic random seed.
            auto_remove_bg: Whether to run background removal.
            num_chunks: SIMD chunk size for SDF queries.

        Returns:
            Tuple of (trimesh.Trimesh, timing_metrics_dict).
        """
        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)

        timing: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. Preprocess reference image
        t_prep_start = time.perf_counter()
        image_tensor, _, comp_rgba = self.preprocessor.preprocess(
            image, auto_remove_bg=auto_remove_bg, dtype=self.dtype, return_rgba=True
        )
        timing["preprocess_s"] = time.perf_counter() - t_prep_start

        # 2. Vision Encoding (DINOv2)
        t_enc_start = time.perf_counter()
        cond = self.image_encoder(image_tensor, value_range=(-1.0, 1.0))
        mx.eval(cond)
        timing["encode_s"] = time.perf_counter() - t_enc_start

        do_cfg = guidance_scale > 0.0
        if do_cfg:
            uncond = self.image_encoder.unconditional_embedding(1)

        # 3. Initialize shape latents (1, 4096, 64)
        num_latents, latent_dim = self.vae.latent_shape
        latents = mx.random.normal((1, num_latents, latent_dim), dtype=self.dtype)

        # 4. Configure Flow-Matching scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 5. Denoising loop
        t_dit_start = time.perf_counter()
        for i in range(num_inference_steps):
            t_val = self.scheduler.timesteps[i]

            if do_cfg:
                latent_input = mx.concatenate([latents, latents], axis=0)
                context = mx.concatenate([cond, uncond], axis=0)
                t_batch = mx.array([t_val, t_val], dtype=self.dtype)
            else:
                latent_input = latents
                context = cond
                t_batch = mx.array([t_val], dtype=self.dtype)

            # Normalize timesteps to [0, 1] range (matching Hunyuan3D-2.1 training)
            t_batch = (t_batch / self.scheduler.num_train_timesteps).astype(self.dtype)

            noise_pred = self.dit(latent_input, t_batch, {"main": context})
            mx.eval(noise_pred)

            if do_cfg:
                pred_cond, pred_uncond = noise_pred[:1], noise_pred[1:]
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # Euler update: x_{t+dt} = x_t + dt * v_t
            latents = self.scheduler.step(noise_pred, latents).astype(self.dtype)
            mx.eval(latents)
            del noise_pred
            if (i + 1) % 5 == 0 and hasattr(mx, "clear_cache"):
                mx.clear_cache()

        timing["dit_s"] = time.perf_counter() - t_dit_start

        # Free DiT conditioning tensors before heavy VAE octree marching cubes
        if do_cfg:
            del uncond
        del cond
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        # 6. Hierarchical SDF volume decode and marching cubes
        t_vae_start = time.perf_counter()
        raw_mesh = self.vae.decode_to_mesh(
            latents,
            bounds=bounds,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            mc_level=mc_level,
        )
        timing["vae_decode_s"] = time.perf_counter() - t_vae_start

        del latents
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        # 7. Mesh cleanup and decimation
        t_clean_start = time.perf_counter()
        cleaned = clean_mesh(raw_mesh)
        final_mesh = simplify_mesh(cleaned, target_faces=target_face_count)
        if comp_rgba is not None:
            final_mesh.metadata["reference_image"] = comp_rgba
        timing["simplify_s"] = time.perf_counter() - t_clean_start

        timing["total_s"] = time.perf_counter() - t0
        return final_mesh, timing

    @classmethod
    def from_pretrained(
        cls,
        weights_source: str = "dgrauet/hunyuan3d-2.1-mlx",
        dtype: mx.Dtype = mx.float16,
    ) -> "Hunyuan3DShapePipeline":
        """Loads all components from local directory or Hugging Face repository."""
        env_override = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
        if env_override:
            weights_source = env_override

        if os.path.isdir(weights_source):
            model_dir = Path(weights_source)
        else:
            from huggingface_hub import snapshot_download

            model_dir = Path(
                snapshot_download(
                    repo_id=weights_source,
                    allow_patterns=[
                        "config.json",
                        "image_encoder.safetensors",
                        "dit.safetensors",
                        "vae.safetensors",
                    ],
                )
            )

        with open(model_dir / "config.json") as f:
            raw_config = json.load(f)

        config = Hunyuan3DConfig.from_dict(raw_config)

        # 1. Load DINOv2
        logger.info(f"Loading DINOv2 from {model_dir / 'image_encoder.safetensors'}...")
        image_encoder = ImageEncoder.from_pretrained(
            str(model_dir / "image_encoder.safetensors"),
            config.__dict__["image_encoder"].__dict__,
        )

        # 2. Load Shape DiT
        logger.info(f"Loading Shape DiT from {model_dir / 'dit.safetensors'}...")
        dit = HunYuanDiTPlain.from_pretrained(
            str(model_dir / "dit.safetensors"),
            config.__dict__["dit"].__dict__,
        )

        # 3. Load ShapeVAE
        logger.info(f"Loading ShapeVAE from {model_dir / 'vae.safetensors'}...")
        vae = ShapeVAEDecoder.from_pretrained(
            str(model_dir / "vae.safetensors"),
            config.__dict__["vae"].__dict__,
        )

        scheduler = FlowMatchEulerScheduler(num_train_timesteps=1000)

        return cls(
            image_encoder=image_encoder,
            dit=dit,
            vae=vae,
            scheduler=scheduler,
            dtype=dtype,
        )
