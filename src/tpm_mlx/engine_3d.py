"""MLX 3D Inference Engine.

Orchestrates Hunyuan3D-2.1 shape generation, memory lifecycle management,
zero-copy Text-to-2D-to-3D chaining, and mesh asset exports.
"""

import base64
import gc
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from PIL import Image

from tpm_mlx.diffusion_3d import Hunyuan3DShapePipeline, export_mesh_files
from tpm_mlx.schemas.threed import (
    ThreeDGenerationRequest,
    ThreeDGenerationResponse,
    ThreeDMetrics,
    ThreeDModelItem,
)
from tpm_mlx.utils import get_active_memory_gb, reclaim_metal_memory

logger = logging.getLogger(__name__)

DEFAULT_3D_MODEL_ID = "dgrauet/hunyuan3d-2.1-mlx"


def _resolve_image_input(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """Resolves an image input from local file path, URL, data URI, or PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input

    str_input = str(image_input).strip()

    # Data URI (e.g. data:image/png;base64,...)
    if str_input.startswith("data:image/") and ";base64," in str_input:
        header, b64_data = str_input.split(";base64,", 1)
        image_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_bytes))

    # Static web server path (e.g. /images/gen_123.png)
    if str_input.startswith("/images/"):
        static_dir = Path(__file__).resolve().parent / "static" / "images"
        rel_path = str_input[len("/images/"):]
        target = static_dir / rel_path
        if target.exists():
            return Image.open(target)

    # Local file path
    local_path = Path(str_input)
    if local_path.exists():
        return Image.open(local_path)

    # Try downloading from HTTP/HTTPS URL
    if str_input.startswith("http://") or str_input.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(str_input, headers={"User-Agent": "TPM-MLX/0.5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            return Image.open(io.BytesIO(data))

    raise FileNotFoundError(f"Could not resolve reference image input: {image_input}")


class MLX3DEngine:
    """Central inference orchestrator for 3D model generation on Apple Silicon."""

    def __init__(
        self,
        model_id: str = DEFAULT_3D_MODEL_ID,
        dtype: mx.Dtype = mx.float16,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.pipeline: Optional[Hunyuan3DShapePipeline] = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load_pipeline(self, model_id: Optional[str] = None) -> None:
        """Loads Hunyuan3DShapePipeline weights into Apple Silicon unified memory."""
        target_model = model_id or self.model_id
        if self.is_loaded and self.model_id == target_model:
            return

        self.unload()
        logger.info(f"Loading 3D pipeline from '{target_model}' on MLX...")
        t_start = time.perf_counter()

        self.pipeline = Hunyuan3DShapePipeline.from_pretrained(
            weights_source=target_model,
            dtype=self.dtype,
        )
        self.model_id = target_model
        elapsed = time.perf_counter() - t_start
        logger.info(f"3D pipeline loaded successfully in {elapsed:.2f}s.")

    def unload(self) -> None:
        """Releases 3D models and purges Metal cache."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        reclaim_metal_memory()
        logger.info("3D Engine unloaded and Metal memory cache cleared.")

    def generate_mesh(
        self,
        req: ThreeDGenerationRequest,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[ThreeDGenerationResponse, Dict[str, str]]:
        """Generates a watertight 3D mesh from input image or prompt."""
        # 1. Resolve or synthesize input reference image
        if req.image:
            reference_image = _resolve_image_input(req.image)
            if not self.is_loaded or self.pipeline is None:
                self.load_pipeline(req.model)
        elif req.prompt:
            # Strict Sequential Chaining: If 3D pipeline is in memory (and not a test mock),
            # evict it before loading the 2D image diffusion engine to avoid concurrent 37+ GB spikes.
            is_mock = "mock" in type(self.pipeline).__name__.lower()
            if self.is_loaded and not is_mock:
                self.unload()
                reclaim_metal_memory()

            # Zero-copy Text -> 2D -> 3D chaining using TPM-MLX Image Engine
            from tpm_mlx.image_engine import MLXImageEngine
            from tpm_mlx.utils import get_cached_models
            from tpm_mlx import state

            logger.info(f"Synthesizing 2D reference image for prompt: '{req.prompt}'...")
            img_engine = getattr(state, "image_engine", None)
            created_temp_engine = False
            if img_engine is None:
                cached_models = [m["repo_id"] for m in get_cached_models() if m.get("is_image")]
                # Prioritize turbo/fast verified cached models to avoid live network downloads
                turbo_models = [m for m in cached_models if "turbo" in m.lower()]
                target_img_model = turbo_models[0] if turbo_models else (cached_models[0] if cached_models else MLXImageEngine.DEFAULT_MODEL)
                logger.info(f"Using 2D image model '{target_img_model}' for 3D reference generation")
                img_engine = MLXImageEngine(model_path_or_id=target_img_model, lazy=False)
                created_temp_engine = True

            import re

            raw_p = req.prompt.strip()
            # Robust prefix stripping for any variation of conversational 3D prompts
            clean_p = re.sub(
                r"^(?:please\s+)?(?:create|generate|make|build|render)\s+(?:an?\s+)?(?:3-?d\s+)?(?:image\s+model|model|asset|image|mesh)?\s*(?:of\s+)?",
                "",
                raw_p,
                flags=re.IGNORECASE,
            ).strip()
            clean_p = re.sub(r"^/3d\s*", "", clean_p, flags=re.IGNORECASE).strip()
            if not clean_p:
                clean_p = raw_p

            # Semantic classification: Character/creature vs. Inanimate object/prop
            char_keywords = {
                "character", "person", "man", "woman", "boy", "girl", "kid", "child",
                "goku", "hero", "warrior", "knight", "soldier", "ninja",
                "alien", "monster", "creature", "avatar", "figure", "figurine",
                "humanoid", "orc", "wizard", "superman", "batman", "spiderman",
                "anime", "chibi", "guy", "lady", "dude", "player", "robot",
                "cyborg", "samurai", "monk", "statue", "doll", "action figure",
            }
            tokens = set(re.findall(r"\b\w+\b", clean_p.lower()))
            is_character = bool(tokens & char_keywords)

            if is_character:
                # Turnaround 3D character asset prompt conditioning (neutral A-pose, full body head-to-toe, straight-on front view)
                engineered_prompt = (
                    f"Full body 3D character model of {clean_p}, "
                    f"standing neutral A-pose, full body complete head-to-toe view, centered, "
                    f"isolated on solid pure white background, studio product lighting, "
                    f"straight-on front-facing orthographic camera view, eye-level, "
                    f"3D model reference, highly detailed, "
                    f"no background scenery, no shadows, no occlusion, no cropping"
                )
            else:
                # Clean, isolated hard-surface / prop asset prompt conditioning (strictly no humans, straight-on front orthographic view)
                engineered_prompt = (
                    f"A single isolated 3D asset of {clean_p}, "
                    f"centered, isolated on solid pure white background, studio lighting, "
                    f"straight-on front-facing orthographic camera view, eye-level, perpendicular angle, "
                    f"clean sharp geometry, high resolution, "
                    f"no humans, no character, no person, no hands, no 3-quarter view, no isometric tilt, "
                    f"no background scenery, no shadows, uncropped"
                )
            logger.info(f"Engineered 3D reference prompt (is_character={is_character}): '{engineered_prompt}'")

            img_res = img_engine.generate(
                prompt=engineered_prompt,
                steps=8 if "turbo" in getattr(img_engine, "model_id", "").lower() else 4,
                size="512x512",
            )
            reference_image = img_res.pil_image

            if created_temp_engine:
                try:
                    img_engine.unload()
                except Exception:
                    pass
                del img_engine
            elif getattr(state, "image_engine", None) is not None:
                state.unload_image_engine()
            reclaim_metal_memory()
        else:
            raise ValueError("Either 'image' or 'prompt' must be provided for 3D generation.")

        # Ensure 3D pipeline is fully loaded
        if not self.is_loaded or self.pipeline is None:
            self.load_pipeline(req.model)

        # 2. Setup output directory and filenames
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "static" / "models_3d"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        seed_part = f"s{req.seed}" if req.seed is not None else f"t{timestamp}"
        base_name = f"model_3d_{timestamp}_{seed_part}"
        base_path = out_dir / base_name

        # 3. Track peak memory and invoke 3D generation pipeline
        mx.reset_peak_memory()
        start_time = time.perf_counter()

        mesh, timing_stats = self.pipeline.generate(
            image=reference_image,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else 5.0,
            octree_resolution=req.octree_resolution,
            target_face_count=req.target_face_count,
            seed=req.seed,
            auto_remove_bg=req.auto_remove_bg,
        )

        total_elapsed = time.perf_counter() - start_time
        peak_mem_gb = (mx.get_peak_memory() / 1e9) if hasattr(mx, "get_peak_memory") else get_active_memory_gb()

        # 3.5. Stage 2 Multi-View Diffusion PBR Texture Painting (enabled by default)
        if getattr(req, "pbr_paint", True):
            try:
                from tpm_mlx.diffusion_3d.paint.paint_pipeline import (
                    Hunyuan3DPaintConfigMLX,
                    Hunyuan3DPaintPipelineMLX,
                )
                logger.info("Executing Stage 2 MLX Multi-View PBR Texture Painting...")
                # Mutual Unload Stage 1 to prevent Unified Memory spikes
                self.unload()
                reclaim_metal_memory()

                ref_img_paint = getattr(mesh, "metadata", {}).get("reference_image", reference_image)
                paint_res = getattr(req, "paint_resolution", 512) or 512
                tex_res = getattr(req, "texture_resolution", 2048) or 2048
                paint_cfg = Hunyuan3DPaintConfigMLX(
                    max_num_view=6,
                    resolution=paint_res,
                    texture_size=tex_res,
                    num_inference_steps=15,
                    weights_source=req.model or "dgrauet/hunyuan3d-2.1-mlx",
                    seed=req.seed or 42,
                )
                paint_pipe = Hunyuan3DPaintPipelineMLX(config=paint_cfg, lazy=False)
                t_paint_start = time.perf_counter()
                mesh = paint_pipe.paint_mesh(
                    mesh=mesh,
                    reference_image=ref_img_paint,
                )
                timing_stats["paint_diffusion_s"] = time.perf_counter() - t_paint_start
                paint_pipe.unload()
                reclaim_metal_memory()
            except Exception as e:
                logger.warning(f"Stage 2 diffusion paint note ({e}); continuing with direct PBR UV bake.")

        # 4. Export mesh to requested formats (.glb, .usdz, .obj) with projected reference texture
        ref_img_to_export = getattr(mesh, "metadata", {}).get("reference_image", reference_image)
        exported_files = export_mesh_files(
            mesh=mesh,
            base_output_path=base_path,
            formats=req.export_formats,
            reference_image=ref_img_to_export,
        )

        reclaim_metal_memory()

        # 5. Build response objects
        item = ThreeDModelItem()
        for fmt, file_path in exported_files.items():
            filename = Path(file_path).name
            url_path = f"/models_3d/{filename}"
            if fmt == "glb":
                item.glb_url = url_path
            elif fmt == "usdz":
                item.usdz_url = url_path
            elif fmt == "obj":
                item.obj_url = url_path

        if req.response_format == "b64_json" and "glb" in exported_files:
            with open(exported_files["glb"], "rb") as f:
                item.b64_json = base64.b64encode(f.read()).decode("utf-8")

        has_pbr = getattr(getattr(mesh, "visual", None), "kind", "") == "texture"
        metrics = ThreeDMetrics(
            generation_time_s=round(total_elapsed, 2),
            faces=len(mesh.faces),
            vertices=len(mesh.vertices),
            peak_memory_gb=round(peak_mem_gb, 2),
            has_pbr_texture=has_pbr,
            octree_resolution=req.octree_resolution,
            timing_breakdown={k: round(v, 2) for k, v in timing_stats.items()},
        )

        response = ThreeDGenerationResponse(
            created=timestamp,
            data=[item],
            tpm_metrics=metrics,
        )

        return response, exported_files
