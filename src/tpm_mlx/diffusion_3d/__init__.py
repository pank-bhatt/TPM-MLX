"""Native Apple Silicon 3D Diffusion subsystem for TPM-MLX."""

from .config import Hunyuan3DConfig, ImageEncoderConfig, ShapeDiTConfig, ShapeVAEConfig
from .dinov2_mlx import ImageEncoder
from .mesh_extract import clean_mesh, export_mesh_files, simplify_mesh
from .pipeline import Hunyuan3DShapePipeline
from .preprocessor import ImageProcessorV2
from .shape_dit import HunYuanDiTPlain
from .shape_vae import ShapeVAEDecoder

__all__ = [
    "Hunyuan3DConfig",
    "ImageEncoderConfig",
    "ShapeDiTConfig",
    "ShapeVAEConfig",
    "ImageProcessorV2",
    "ImageEncoder",
    "HunYuanDiTPlain",
    "ShapeVAEDecoder",
    "Hunyuan3DShapePipeline",
    "clean_mesh",
    "simplify_mesh",
    "export_mesh_files",
]
