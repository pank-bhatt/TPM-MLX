"""Hunyuan3D-2.1 Stage 2 PBR Paint Pipeline for Apple Silicon MLX.

Orchestrates:
1. xatlas UV unwrapping.
2. Metal multi-view rendering (normal maps, position maps) via MeshRenderMLX.
3. Multi-view 2.5D diffusion UNet (HunyuanPaintModelMLX) with DINOv2 and reference cross-attention.
4. (Optional) RealESRGAN x4 super-resolution on MLX.
5. Cosine-weighted multi-view texture baking into 4096² UV atlas (Albedo + Roughness/Metallic).
6. EDT gutter inpainting and packaging into standard glTF PBRMaterial GLB.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlx.core as mx
import numpy as np
import trimesh
from PIL import Image

from tpm_mlx.diffusion_3d.renderer.mesh_render_mlx import MeshRenderMLX
from tpm_mlx.diffusion_3d.renderer.pipeline_utils_mlx import ViewProcessorMLX
from tpm_mlx.diffusion_3d.renderer.uvwrap_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DPaintConfigMLX:
    """Configuration for MLX-native Hunyuan3D-2.1 Paint Pipeline."""

    def __init__(
        self,
        max_num_view: int = 6,
        resolution: int = 512,
        texture_size: int = 2048,
        num_inference_steps: int = 15,
        guidance_scale: float = 3.0,
        weights_source: str = "dgrauet/hunyuan3d-2.1-mlx",
        use_super_res: bool = False,
        seed: Optional[int] = 42,
    ):
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.render_size = resolution
        self.texture_size = texture_size
        self.mlx_num_inference_steps = num_inference_steps
        self.mlx_guidance_scale = guidance_scale
        self.mlx_weights_source = weights_source
        self.mlx_seed = seed
        self.use_mlx_super_res = use_super_res
        self.mlx_super_res_filename = "realesrgan_x4plus.safetensors"

        self.raster_mode = "mlx"
        self.bake_mode = "back_sample"
        self.bake_exp = 4
        self.merge_method = "fast"

        # Candidate camera viewpoints for greedy set-cover selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1.0, 0.5, 0.6, 0.5, 0.4, 0.2]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipelineMLX:
    """Full MLX Stage 2 texture generation pipeline."""

    def __init__(self, config: Optional[Hunyuan3DPaintConfigMLX] = None, lazy: bool = True):
        self.config = config or Hunyuan3DPaintConfigMLX()
        self.models: Dict[str, object] = {}
        self.render = MeshRenderMLX(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
        )
        self.view_processor = ViewProcessorMLX(self.config, self.render)
        self.is_loaded = False

        if not lazy:
            self.load_models()

    def load_models(self) -> None:
        """Loads MLX multi-view diffusion UNet, VAE, and DINOv2 weights."""
        if self.is_loaded:
            return

        from tpm_mlx.diffusion_3d.paint.hunyuanpaintpbr_mlx.load_model import HunyuanPaintModelMLX

        logger.info(f"Loading Stage 2 MLX Paint model from '{self.config.mlx_weights_source}'...")
        self.models["mlx_model"] = HunyuanPaintModelMLX.from_pretrained(
            self.config.mlx_weights_source
        )

        if getattr(self.config, "use_mlx_super_res", False):
            try:
                from tpm_mlx.diffusion_3d.paint.image_super_utils_mlx import imageSuperNetMLX
                sr_path = self._resolve_super_res_weights()
                self.models["mlx_super"] = imageSuperNetMLX(sr_path)
                logger.info("RealESRGAN super-resolution enabled on MLX.")
            except Exception as e:
                logger.warning(f"Super-resolution unavailable ({e}); proceeding without.")

        self.is_loaded = True
        logger.info("Stage 2 MLX Paint pipeline loaded successfully.")

    def unload(self) -> None:
        """Evicts paint models and purges Metal memory."""
        self.models.clear()
        self.is_loaded = False
        from tpm_mlx.utils import reclaim_metal_memory
        reclaim_metal_memory()
        logger.info("Stage 2 Paint Pipeline unloaded and Metal cache cleared.")

    def _resolve_super_res_weights(self) -> str:
        src = self.config.mlx_weights_source
        filename = self.config.mlx_super_res_filename
        env_override = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
        if env_override:
            return os.path.join(env_override, filename)
        if os.path.isdir(src):
            return os.path.join(src, filename)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=src, filename=filename)

    def _run_multiview_mlx(
        self,
        image_style: List[Image.Image],
        normal_maps: List[Image.Image],
        position_maps: List[Image.Image],
        camera_azims: List[float],
    ) -> Dict[str, List[Image.Image]]:
        from tpm_mlx.diffusion_3d.paint.hunyuanpaintpbr_mlx.inference import generate_multiview_pbr

        ref_np = np.array(image_style[0].convert("RGB"))

        def _pil_to_np_f32(img):
            return np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        n_maps = [_pil_to_np_f32(m) for m in normal_maps]
        p_maps = [_pil_to_np_f32(m) for m in position_maps]

        result = generate_multiview_pbr(
            model=self.models["mlx_model"],
            normal_maps=n_maps,
            position_maps=p_maps,
            reference_image=ref_np,
            camera_azims=camera_azims,
            num_inference_steps=self.config.mlx_num_inference_steps,
            guidance_scale=self.config.mlx_guidance_scale,
            view_size=self.config.resolution,
            seed=self.config.mlx_seed,
        )

        return {
            "albedo": [Image.fromarray(a) for a in result["albedo"]],
            "mr": [Image.fromarray(m) for m in result["mr"]],
        }

    def paint_mesh(
        self,
        mesh: trimesh.Trimesh,
        reference_image: Union[str, Path, Image.Image],
        output_glb_path: Optional[Union[str, Path]] = None,
    ) -> trimesh.Trimesh:
        """Paints 3D mesh using multi-view diffusion and bakes full PBR textures."""
        if not self.is_loaded:
            self.load_models()

        if isinstance(reference_image, (str, Path)):
            ref_pil = Image.open(str(reference_image))
        elif isinstance(reference_image, Image.Image):
            ref_pil = reference_image
        else:
            ref_pil = Image.fromarray(reference_image)

        # 1. Prepare reference image with clean background
        ref_rgba = ref_pil.convert("RGBA")
        white_bg = Image.new("RGB", ref_rgba.size, (255, 255, 255))
        if "A" in ref_rgba.getbands():
            white_bg.paste(ref_rgba, mask=ref_rgba.getchannel("A"))
        else:
            white_bg.paste(ref_rgba)

        image_style = [white_bg.resize((512, 512))]

        # 2. UV unwrap mesh via xatlas
        if getattr(getattr(mesh, "visual", None), "uv", None) is None or len(mesh.visual.uv) == 0:
            mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh=mesh)

        # 3. View Selection via greedy set cover
        selected_elevs, selected_azims, selected_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        # 4. Render conditioning maps (Metal GPU)
        normal_maps = self.view_processor.render_normal_multiview(
            selected_elevs, selected_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(
            selected_elevs, selected_azims
        )

        # 5. Run multi-view diffusion synthesis (MLX)
        multiviews_pbr = self._run_multiview_mlx(
            image_style, normal_maps, position_maps, selected_azims
        )
        enhance_images = {
            "albedo": list(multiviews_pbr["albedo"]),
            "mr": list(multiviews_pbr["mr"]),
        }

        # Optional RealESRGAN super-resolution
        if "mlx_super" in self.models:
            sr = self.models["mlx_super"]
            for i in range(len(enhance_images["albedo"])):
                enhance_images["albedo"][i] = sr(enhance_images["albedo"][i])
                enhance_images["mr"][i] = sr(enhance_images["mr"][i])

        # 6. Texture baking (Metal GPU)
        render_size = self.config.render_size
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (render_size, render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                (render_size, render_size)
            )

        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_elevs, selected_azims, selected_weights
        )
        mask_np = (mask[..., 0] * 255).astype(np.uint8)

        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_elevs, selected_azims, selected_weights
        )
        mask_mr_np = (mask_mr[..., 0] * 255).astype(np.uint8)

        # 7. Inpainting and gutter fill (OpenCV + EDT)
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture)

        texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
        self.render.set_texture_mr(texture_mr)

        # 8. Build PBR textured mesh
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".obj") as tmp_obj:
            self.render.save_mesh(tmp_obj.name, downsample=False)

        final_mesh = getattr(self.render, "_last_exported_mesh", mesh)

        if output_glb_path is not None:
            Path(output_glb_path).parent.mkdir(parents=True, exist_ok=True)
            final_mesh.export(str(output_glb_path), file_type="glb")
            logger.info(f"PBR textured GLB saved to {output_glb_path}")

        return final_mesh
