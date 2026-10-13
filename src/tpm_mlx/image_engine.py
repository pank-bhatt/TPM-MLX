"""
TPM-MLX: High-Performance Apple Silicon Text-to-Image Inference Engine.

Provides native MLX diffusion and flow-matching image generation (FLUX.2, Bonsai,
Z-Image) with memory tracking, deterministic seeds, and OpenAI-compatible outputs.
"""

from __future__ import annotations

import base64
import gc
import logging
import random
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import mlx.core as mx
from PIL import Image

from tpm_mlx.utils import get_active_memory_gb

logger = logging.getLogger("tpm-mlx.image_engine")


@dataclass
class ImageGenerationMetrics:
    """Performance telemetry for a text-to-image generation run."""
    generation_time_s: float = 0.0
    steps: int = 4
    step_time_s: float = 0.0
    peak_memory_gb: float = 0.0
    width: int = 512
    height: int = 512
    seed: int = 0
    model: str = ""
    supports_editing: bool = False


@dataclass
class ImageResult:
    """Result of an image generation operation."""
    pil_image: Image.Image
    b64_json: str
    png_bytes: bytes
    metrics: ImageGenerationMetrics
    saved_path: Optional[Path] = None
    revised_prompt: Optional[str] = None



def _resolve_input_image_path(image_ref: Union[str, Path]) -> Path:
    """Resolves an image reference (URL, relative path, base64 data, or absolute path) to a local file securely."""
    ref_str = str(image_ref).strip()
    static_dir = (Path(__file__).parent / "static" / "images").resolve()

    # Handle Base64 Data URL (e.g. data:image/png;base64,...)
    if ref_str.startswith("data:image"):
        parts = ref_str.split(",", 1)
        encoded = parts[1] if len(parts) > 1 else parts[0]
        data = base64.b64decode(encoded)
        uploads_dir = (static_dir / "uploads").resolve()
        uploads_dir.mkdir(parents=True, exist_ok=True)
        out_file = uploads_dir / f"ref_{int(time.time())}_{random.randint(1000, 9999)}.png"
        out_file.write_bytes(data)
        return out_file

    # Handle relative static images URL (e.g. /images/gen_123.png)
    if ref_str.startswith("/images/"):
        rel_subpath = ref_str.removeprefix("/images/").lstrip("/")
        static_img = (static_dir / rel_subpath).resolve()
        if static_img.is_relative_to(static_dir) and static_img.is_file():
            return static_img
        raise PermissionError(f"Access denied or file not found: '{ref_str}'")

    # Handle standard path
    p = Path(ref_str).expanduser().resolve()
    if p.is_file():
        return p

    # Check relative to static/images
    alt_p = (static_dir / ref_str.lstrip("/")).resolve()
    if alt_p.is_relative_to(static_dir) and alt_p.is_file():
        return alt_p

    raise FileNotFoundError(f"Reference image not found: '{ref_str}'")


def _register_qwen_turbo_scheduler() -> None:
    """
    Registers the 8-step distilled Qwen21TurboScheduler into mflux, enables it for editing,
    and installs a transparent weight-loader adapter for mlx-community/Qwen-Image-2.1-MLX-4bit.
    """
    try:
        from mlx.utils import tree_unflatten
        from mflux.models.common.schedulers import register_contrib
        from mflux.models.common.schedulers.base_scheduler import BaseScheduler
        from mflux.models.common.schedulers.linear_scheduler import LinearScheduler
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
        import mflux.models.qwen21.variants.edit.qwen_image_21_edit as qwen_edit_mod

        class Qwen21TurboScheduler(BaseScheduler):
            """Official 8-step distilled sigma schedule for Qwen-Image-2.1-Turbo."""
            SIGMA_NODES = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)

            def __init__(self, config):
                self.config = config
                self._sigmas = LinearScheduler.shift_sigmas(
                    mx.array(list(self.SIGMA_NODES), dtype=mx.float32), config, use_terminal=False
                )
                self._timesteps = mx.arange(len(self.SIGMA_NODES), dtype=mx.float32)

            @property
            def sigmas(self) -> mx.array:
                return self._sigmas

            @property
            def timesteps(self) -> mx.array:
                return self._timesteps

            def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
                dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
                return latents + noise.astype(latents.dtype) * dt

        register_contrib(Qwen21TurboScheduler, "qwen_turbo")
        register_contrib(Qwen21TurboScheduler, "Qwen21TurboScheduler")

        if "qwen_turbo" not in qwen_edit_mod.EDIT_SCHEDULERS:
            qwen_edit_mod.EDIT_SCHEDULERS = (*qwen_edit_mod.EDIT_SCHEDULERS, "qwen_turbo")

        # Install adapter for mlx-community/Qwen-Image-2.1-MLX-4bit (which uses {'format': 'mlx'} metadata)
        if not getattr(WeightLoader, "_tpm_mlx_qwen21_patched", False):
            orig_load_single_local = WeightLoader.load_single_local

            def _patched_load_single_local(component, root_path: Path) -> LoadedWeights:
                sub_path = root_path / component.hf_subdir
                shard_files = sorted(f for f in sub_path.glob("*.safetensors") if not f.name.startswith("._")) if sub_path.exists() else []
                if shard_files:
                    _, meta = mx.load(str(shard_files[0]), return_metadata=True)
                    if meta and meta.get("format") == "mlx" and "mflux_version" not in meta:
                        all_weights = {}
                        for shard in shard_files:
                            w, _ = mx.load(str(shard), return_metadata=True)
                            all_weights.update(dict(w.items()))

                        if component.name == "vae":
                            return LoadedWeights(
                                components={"vae": tree_unflatten(list(all_weights.items()))},
                                meta_data=MetaData(quantization_level=None, mflux_version="mlx-community"),
                            )
                        elif component.name == "transformer":
                            norm_t = {}
                            remap_dequant = {
                                "modulation.0": "modulation.layers.1",
                                "norm_out.linear": "norm_out.linear",
                                "time_text_embed.linear_1": "time_text_embed.timestep_embedder.linear_1",
                                "time_text_embed.linear_2": "time_text_embed.timestep_embedder.linear_2",
                            }
                            handled = set()
                            for src_p, dst_p in remap_dequant.items():
                                if f"{src_p}.scales" in all_weights:
                                    dq = mx.dequantize(
                                        all_weights[f"{src_p}.weight"],
                                        all_weights[f"{src_p}.scales"],
                                        all_weights[f"{src_p}.biases"],
                                        group_size=64,
                                        bits=4,
                                    )
                                    norm_t[f"{dst_p}.weight"] = dq
                                    handled.update({f"{src_p}.weight", f"{src_p}.scales", f"{src_p}.biases"})
                            for k, v in all_weights.items():
                                if k not in handled:
                                    norm_t[k] = v
                            return LoadedWeights(
                                components={"transformer": tree_unflatten(list(norm_t.items()))},
                                meta_data=MetaData(quantization_level=4, mflux_version="mlx-community"),
                            )
                        elif component.name == "text_encoder":
                            norm_te = {}
                            for k, v in all_weights.items():
                                nk = (
                                    k.replace("language_model.model.", "language_model.")
                                    .replace("language_model.lm_head.", "lm_head.")
                                    .replace("vision_tower.", "visual.")
                                )
                                if nk.endswith("patch_embed.proj.weight") and v.ndim == 5 and v.shape[1] == 3:
                                    v = v.transpose(0, 2, 3, 4, 1)
                                norm_te[nk] = v
                            return LoadedWeights(
                                components={"text_encoder": tree_unflatten(list(norm_te.items()))},
                                meta_data=MetaData(quantization_level=4, mflux_version="mlx-community"),
                            )
                return orig_load_single_local(component, root_path)

            WeightLoader.load_single_local = staticmethod(_patched_load_single_local)
            WeightLoader._tpm_mlx_qwen21_patched = True
    except Exception as e:
        logger.debug(f"Could not register Qwen21TurboScheduler: {e}")


_register_qwen_turbo_scheduler()


class MLXImageEngine:
    """
    Dedicated Text-to-Image and Image-to-Image Editing Engine powered by native Apple Silicon MLX.
    Supports FLUX.2 (Klein 4B/9B), Qwen-Image-2.1 (Turbo 8-step & Base 40-step), Bonsai (2-bit/ternary),
    and Z-Image flow models.
    """

    DEFAULT_MODEL = "flux2-klein-4b"

    def __init__(
        self,
        model_path_or_id: Optional[str] = None,
        lazy: bool = False,
    ):
        self.model_id = model_path_or_id or self.DEFAULT_MODEL
        self.model: Any = None
        self.edit_model: Any = None
        self.qwen21_model: Any = None
        self.qwen21_edit_model: Any = None
        self.is_loaded: bool = False

        if not lazy:
            self._load_model()

    @staticmethod
    def _is_qwen21(model_id: str) -> bool:
        """Returns True if the model ID refers to a Qwen-Image-2.1 variant."""
        lower = (model_id or "").lower()
        return ("qwen" in lower and "image" in lower) or "studio-image" in lower

    @property
    def is_qwen21(self) -> bool:
        """Returns True if this engine instance is configured for Qwen-Image-2.1."""
        return self._is_qwen21(self.model_id)

    @property
    def is_turbo(self) -> bool:
        """Returns True if this model uses a distilled Turbo schedule."""
        return "turbo" in self.model_id.lower()

    @property
    def supports_editing(self) -> bool:
        """Returns True if the model supports image-to-image editing."""
        lower = self.model_id.lower()
        return "flux" in lower or self.is_qwen21


    def _load_qwen21_model(self) -> None:
        """Loads Qwen-Image-2.1 via native mflux unified QwenImage21Edit backend."""
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        logger.info(f"Loading Qwen-Image-2.1 model via mflux: '{self.model_id}'...")
        start_t = time.perf_counter()

        # Pass quantize=4 if 4bit/q4 is in model_id and checkpoint isn't pre-quantized
        lower = self.model_id.lower()
        quantize = 4 if ("4bit" in lower or "q4" in lower) else (8 if ("8bit" in lower or "q8" in lower) else None)
        self.qwen21_edit_model = QwenImage21Edit(model_path=self.model_id, quantize=quantize)
        self.qwen21_model = self.qwen21_edit_model
        self.model = self.qwen21_edit_model
        self.edit_model = self.qwen21_edit_model
        self.is_loaded = True
        elapsed = time.perf_counter() - start_t
        active_mem = get_active_memory_gb()
        logger.info(
            f"Loaded Qwen-Image-2.1 model '{self.model_id}' in {elapsed:.2f}s "
            f"(Active Unified Memory: {active_mem:.2f} GB)"
        )

    def _load_model(self) -> None:
        """Loads the image model into Apple Silicon Unified Memory."""
        if self.is_loaded and self.model is not None:
            return

        if self.is_qwen21:
            self._load_qwen21_model()
            return

        from mlx_vlm.generate.image import load_image_model

        logger.info(f"Loading image generation model: '{self.model_id}'...")
        start_t = time.perf_counter()

        self.model = load_image_model(self.model_id)
        self._eager_eval_pipeline(self.model)
        self.is_loaded = True
        elapsed = time.perf_counter() - start_t
        active_mem = get_active_memory_gb()
        logger.info(f"Loaded image model '{self.model_id}' in {elapsed:.2f}s (Active Unified Memory: {active_mem:.2f} GB)")

    @staticmethod
    def _eager_eval_pipeline(model_obj: Any) -> None:
        """
        Eagerly evaluates model weights and internal position constants into Apple Silicon Unified Memory.
        This prevents thread-bound lazy computation graph references across different execution threads
        (which causes 'RuntimeError: There is no Stream(cpu, 0) in current thread').
        """
        try:
            if hasattr(model_obj, "pipeline"):
                pipe = model_obj.pipeline
                for enc_name in ("text_encoder", "text_encoder_2"):
                    enc = getattr(pipe, enc_name, None)
                    if enc is not None:
                        if hasattr(enc, "parameters"):
                            mx.eval(enc.parameters())
                        if hasattr(enc, "rotary_emb"):
                            for attr in ("_inv_freq", "inv_freq"):
                                if hasattr(enc.rotary_emb, attr):
                                    mx.eval(getattr(enc.rotary_emb, attr))
                if getattr(pipe, "transformer", None) is not None:
                    if hasattr(pipe.transformer, "parameters"):
                        mx.eval(pipe.transformer.parameters())
                if getattr(pipe, "vae", None) is not None:
                    if hasattr(pipe.vae, "parameters"):
                        mx.eval(pipe.vae.parameters())
            elif hasattr(model_obj, "parameters"):
                mx.eval(model_obj.parameters())
            mx.synchronize()
        except Exception as e:
            logger.debug(f"Parameter eager evaluation note: {e}")

    def unload(self) -> None:
        """Unloads the model and reclaims Apple Silicon Metal buffer memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.edit_model is not None:
            del self.edit_model
            self.edit_model = None
        if self.qwen21_model is not None:
            del self.qwen21_model
            self.qwen21_model = None
        if self.qwen21_edit_model is not None:
            del self.qwen21_edit_model
            self.qwen21_edit_model = None
        self.is_loaded = False
        gc.collect()
        mx.clear_cache()
        if hasattr(mx, "metal"):
            mx.metal.clear_cache()
        logger.info("Image engine unloaded and Metal cache cleared.")

    @staticmethod
    def extract_resolution_from_text(text: str) -> Optional[str]:
        """Extracts resolution tags like '1024x1024', '16:9', '1:1' from prompt text."""
        import re
        if not text:
            return None
        # Explicit WxH like 1024x1024, 512x768, 1024*576
        m = re.search(r'\b(\d{3,4})\s*[xX×*]\s*(\d{3,4})\b', text)
        if m:
            return f"{m.group(1)}x{m.group(2)}"
        # Aspect ratios like 16:9, 9:16, 4:3, 3:4, 1:1
        m_ar = re.search(r'\b(16:9|9:16|4:3|3:4|1:1)\b', text)
        if m_ar:
            return m_ar.group(1)
        return None

    @staticmethod
    def parse_dimensions(size: str) -> Tuple[int, int]:
        """Parses resolution strings like '1024x1024', '512x512', or '16:9'."""
        normalized = size.lower().replace("×", "x").replace("*", "x").strip()
        aspect_presets = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "square": (1024, 1024),
            "1k": (1024, 1024),
        }
        if normalized in aspect_presets:
            return aspect_presets[normalized]

        if "x" in normalized:
            parts = normalized.split("x")
            try:
                w, h = int(parts[0]), int(parts[1])
                # Round to multiples of 16 for VAE compatibility
                w = max(64, (w // 16) * 16)
                h = max(64, (h // 16) * 16)
                return w, h
            except ValueError:
                pass

        return 1024, 1024

    def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: int = 4,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        revised_prompt: Optional[str] = None,
        image_paths: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    ) -> ImageResult:
        """
        Executes text-to-image generation (or image-to-image edit if image_paths provided).

        Args:
            prompt: Text description of the image to generate or edit.
            size: Resolution string (e.g. '1024x1024', '512x512', '16:9').
            steps: Number of denoising flow steps (default 4 for distilled models).
            guidance: Guidance scale (optional, defaults to model config).
            seed: Deterministic random seed.
            output_path: Optional file path to save the generated PNG.
            revised_prompt: Optional LLM-distilled prompt from long context.
            image_paths: Optional reference image(s) to edit/modify.
        """
        if image_paths is not None:
            return self.edit(
                image_paths=image_paths,
                prompt=prompt,
                size=size,
                steps=steps,
                guidance=guidance,
                seed=seed,
                output_path=output_path,
                revised_prompt=revised_prompt,
            )

        if not self.is_loaded:
            self._load_model()

        if self.is_qwen21:
            return self._generate_via_qwen21_pipeline(
                prompt=prompt,
                size=size,
                steps=steps,
                guidance=guidance,
                seed=seed,
                output_path=output_path,
                revised_prompt=revised_prompt,
            )

        from mlx_vlm.generate.image import ImageGenerationRequest, generate_image

        width, height = self.parse_dimensions(size)
        effective_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        req = ImageGenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=effective_seed,
        )

        logger.info(f"Generating image: '{prompt[:60]}...' ({width}x{height}, {steps} steps, seed={effective_seed})")
        mx.reset_peak_memory()
        start_time = time.perf_counter()

        with mx.stream(mx.default_stream(mx.default_device())):
            self._eager_eval_pipeline(self.model)
            gen_result = generate_image(self.model, req)
            mx.eval(gen_result.array)

        elapsed = time.perf_counter() - start_time
        peak_mem = get_active_memory_gb()

        pil_img = gen_result.to_pil()
        png_bytes = gen_result.to_png_bytes()
        b64_json = base64.b64encode(png_bytes).decode("ascii")

        saved_file: Optional[Path] = None
        if output_path:
            p = Path(output_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(p, format="PNG")
            saved_file = p

        step_time = elapsed / max(steps, 1)
        metrics = ImageGenerationMetrics(
            generation_time_s=round(elapsed, 2),
            steps=steps,
            step_time_s=round(step_time, 3),
            peak_memory_gb=round(peak_mem, 2),
            width=width,
            height=height,
            seed=effective_seed,
            model=self.model_id,
            supports_editing=self.supports_editing,
        )

        logger.info(f"Image generated in {elapsed:.2f}s ({step_time:.3f}s/step, VRAM: {peak_mem:.2f} GB)")

        return ImageResult(
            pil_image=pil_img,
            b64_json=b64_json,
            png_bytes=png_bytes,
            metrics=metrics,
            saved_path=saved_file,
            revised_prompt=revised_prompt or prompt,
        )

    def _generate_via_qwen21_pipeline(
        self,
        prompt: str,
        size: str = "1024x1024",
        steps: int = 8,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        revised_prompt: Optional[str] = None,
    ) -> ImageResult:
        """Executes text-to-image generation via mflux QwenImage21Edit (unified pipeline)."""
        if self.qwen21_edit_model is None:
            self._load_qwen21_model()

        width, height = self.parse_dimensions(size)
        effective_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        if self.is_turbo:
            scheduler = "qwen_turbo"
            effective_steps = 8
        else:
            scheduler = "linear"
            # If caller passed default 4 steps, auto-upgrade to 40 for base model
            effective_steps = 40 if steps == 4 else steps

        # Qwen-Image-2.1 uses CFG = 1.0 by default (single pass per step)
        effective_guidance = 1.0

        logger.info(
            f"Generating Qwen-Image-2.1 image: '{prompt[:60]}...' "
            f"({width}x{height}, {effective_steps} steps, scheduler={scheduler}, seed={effective_seed})"
        )
        mx.reset_peak_memory()
        start_time = time.perf_counter()

        with mx.stream(mx.default_stream(mx.default_device())):
            generated = self.qwen21_edit_model.generate_image(
                seed=effective_seed,
                prompt=prompt,
                num_inference_steps=effective_steps,
                height=height,
                width=width,
                guidance=effective_guidance,
                image_paths=[],
                use_kv_cache=True,
                scheduler=scheduler,
            )

        elapsed = time.perf_counter() - start_time
        peak_mem = get_active_memory_gb()

        pil_img = generated.image
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64_json = base64.b64encode(png_bytes).decode("ascii")

        saved_file: Optional[Path] = None
        if output_path:
            p = Path(output_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(p, format="PNG")
            saved_file = p

        step_time = elapsed / max(effective_steps, 1)
        metrics = ImageGenerationMetrics(
            generation_time_s=round(elapsed, 2),
            steps=effective_steps,
            step_time_s=round(step_time, 3),
            peak_memory_gb=round(peak_mem, 2),
            width=width,
            height=height,
            seed=effective_seed,
            model=self.model_id,
            supports_editing=True,
        )

        logger.info(f"Qwen-Image-2.1 generated in {elapsed:.2f}s ({step_time:.3f}s/step, VRAM: {peak_mem:.2f} GB)")

        return ImageResult(
            pil_image=pil_img,
            b64_json=b64_json,
            png_bytes=png_bytes,
            metrics=metrics,
            saved_path=saved_file,
            revised_prompt=revised_prompt or prompt,
        )

    def _edit_via_qwen21_pipeline(
        self,
        resolved_paths: Sequence[Path],
        prompt: str,
        size: Optional[str] = None,
        steps: int = 8,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        revised_prompt: Optional[str] = None,
    ) -> ImageResult:
        """Executes instruction-based image editing via mflux QwenImage21Edit."""
        if self.qwen21_edit_model is None:
            self._load_qwen21_model()

        if size:
            width, height = self.parse_dimensions(size)
        else:
            with Image.open(resolved_paths[0]) as ref_img:
                w, h = ref_img.size
                width = max(64, (w // 16) * 16)
                height = max(64, (h // 16) * 16)

        effective_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        if self.is_turbo:
            scheduler = "qwen_turbo"
            effective_steps = 8
        else:
            scheduler = "linear"
            effective_steps = 40 if steps == 4 else steps

        # QwenImage21Edit requires negative_prompt if guidance > 1.0; default to CFG=1.0
        effective_guidance = 1.0

        logger.info(
            f"Editing Qwen-Image-2.1 image: '{prompt[:60]}...' "
            f"({width}x{height}, {effective_steps} steps, scheduler={scheduler}, seed={effective_seed}, ref={resolved_paths[0].name})"
        )
        mx.reset_peak_memory()
        start_time = time.perf_counter()

        with mx.stream(mx.default_stream(mx.default_device())):
            generated = self.qwen21_edit_model.generate_image(
                seed=effective_seed,
                prompt=prompt,
                num_inference_steps=effective_steps,
                height=height,
                width=width,
                guidance=effective_guidance,
                image_paths=list(resolved_paths),
                use_kv_cache=True,
                scheduler=scheduler,
            )

        elapsed = time.perf_counter() - start_time
        peak_mem = get_active_memory_gb()

        pil_img = generated.image
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64_json = base64.b64encode(png_bytes).decode("ascii")

        saved_file: Optional[Path] = None
        if output_path:
            p = Path(output_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(p, format="PNG")
            saved_file = p

        step_time = elapsed / max(effective_steps, 1)
        metrics = ImageGenerationMetrics(
            generation_time_s=round(elapsed, 2),
            steps=effective_steps,
            step_time_s=round(step_time, 3),
            peak_memory_gb=round(peak_mem, 2),
            width=width,
            height=height,
            seed=effective_seed,
            model=self.model_id,
            supports_editing=True,
        )

        logger.info(f"Qwen-Image-2.1 edited in {elapsed:.2f}s ({step_time:.3f}s/step, VRAM: {peak_mem:.2f} GB)")

        return ImageResult(
            pil_image=pil_img,
            b64_json=b64_json,
            png_bytes=png_bytes,
            metrics=metrics,
            saved_path=saved_file,
            revised_prompt=revised_prompt or prompt,
        )

    def edit(
        self,
        image_paths: Union[str, Path, Sequence[Union[str, Path]]],
        prompt: str,
        size: Optional[str] = None,
        steps: int = 4,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        revised_prompt: Optional[str] = None,
    ) -> ImageResult:
        """
        Modifies or edits existing reference image(s) according to a natural language prompt.

        Args:
            image_paths: Path(s) or reference(s) to existing image(s) to modify.
            prompt: Text instruction describing the desired edits.
            size: Target resolution (optional, defaults to source image size rounded to multiple of 16).
            steps: Denoising flow steps (default 4).
            guidance: Edit guidance strength (defaults to 2.5).
            seed: Deterministic seed.
            output_path: Optional output file path.
            revised_prompt: Optional LLM-distilled edit prompt.
        """
        if not self.is_loaded:
            self._load_model()

        if not self.supports_editing:
            raise ValueError(
                f"Model '{self.model_id}' does not support instruction-based image editing. "
                "Please select an edit-capable model such as 'OsaurusAI/Qwen-Image-2.1-Turbo-mflux-4bit' or 'mlx-community/FLUX.2-klein-9B'."
            )

        # Normalize reference image paths
        raw_paths = [image_paths] if isinstance(image_paths, (str, Path)) else list(image_paths)
        if not raw_paths:
            raise ValueError("At least one reference image path is required for editing.")
        resolved_paths = [_resolve_input_image_path(p) for p in raw_paths]

        if self.is_qwen21:
            return self._edit_via_qwen21_pipeline(
                resolved_paths=resolved_paths,
                prompt=prompt,
                size=size,
                steps=steps,
                guidance=guidance,
                seed=seed,
                output_path=output_path,
                revised_prompt=revised_prompt,
            )

        from mlx_vlm.generate.edit_image import ImageEditRequest, edit_image

        # Prepare edit model
        if self.edit_model is None:
            if hasattr(self.model, "is_image_edit_model") and self.model.is_image_edit_model:
                self.edit_model = self.model
            else:
                from mlx_vlm.generate.image import _resolve_image_model_path
                from mlx_vlm.generate.edit_image import load_image_edit_model
                resolved_p = _resolve_image_model_path(self.model_id)
                logger.info(f"Loading edit pipeline for model '{self.model_id}'...")
                self.edit_model = load_image_edit_model(str(resolved_p))
                self._eager_eval_pipeline(self.edit_model)

        edit_model = self.edit_model

        # Determine canvas dimensions: use specified size or inherit from source image
        if size:
            width, height = self.parse_dimensions(size)
        else:
            with Image.open(resolved_paths[0]) as ref_img:
                w, h = ref_img.size
                width = max(64, (w // 16) * 16)
                height = max(64, (h // 16) * 16)

        effective_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        effective_guidance = guidance if guidance is not None else 2.5

        req = ImageEditRequest(
            prompt=prompt,
            image_paths=tuple(resolved_paths),
            width=width,
            height=height,
            steps=steps,
            guidance=effective_guidance,
            seed=effective_seed,
        )

        logger.info(f"Editing image: '{prompt[:60]}...' ({width}x{height}, {steps} steps, seed={effective_seed}, ref={resolved_paths[0].name})")
        mx.reset_peak_memory()
        start_time = time.perf_counter()

        with mx.stream(mx.default_stream(mx.default_device())):
            self._eager_eval_pipeline(edit_model)
            gen_result = edit_image(edit_model, req)
            mx.eval(gen_result.array)

        elapsed = time.perf_counter() - start_time
        peak_mem = get_active_memory_gb()

        pil_img = gen_result.to_pil()
        png_bytes = gen_result.to_png_bytes()
        b64_json = base64.b64encode(png_bytes).decode("ascii")

        saved_file: Optional[Path] = None
        if output_path:
            p = Path(output_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(p, format="PNG")
            saved_file = p

        step_time = elapsed / max(steps, 1)
        metrics = ImageGenerationMetrics(
            generation_time_s=round(elapsed, 2),
            steps=steps,
            step_time_s=round(step_time, 3),
            peak_memory_gb=round(peak_mem, 2),
            width=width,
            height=height,
            seed=effective_seed,
            model=self.model_id,
            supports_editing=True,
        )

        logger.info(f"Image edited in {elapsed:.2f}s ({step_time:.3f}s/step, VRAM: {peak_mem:.2f} GB)")

        return ImageResult(
            pil_image=pil_img,
            b64_json=b64_json,
            png_bytes=png_bytes,
            metrics=metrics,
            saved_path=saved_file,
            revised_prompt=revised_prompt or prompt,
        )
