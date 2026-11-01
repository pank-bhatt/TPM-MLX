"""Unit test suite for TPM-MLX 3D Generation Subsystem (Stage 1 Geometry)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image, ImageDraw
import trimesh

from tpm_mlx.diffusion_3d.config import (
    Hunyuan3DConfig,
    ImageEncoderConfig,
    ShapeDiTConfig,
    ShapeVAEConfig,
)
from tpm_mlx.diffusion_3d.dinov2_mlx import ImageEncoder
from tpm_mlx.diffusion_3d.mesh_extract import (
    apply_studio_clay_material,
    clean_mesh,
    export_mesh_files,
    simplify_mesh,
)
from tpm_mlx.diffusion_3d.preprocessor import ImageProcessorV2
from tpm_mlx.diffusion_3d.shape_dit import HunYuanDiTPlain
from tpm_mlx.diffusion_3d.shape_vae import ShapeVAEDecoder
from tpm_mlx.schemas.threed import (
    ThreeDGenerationRequest,
    ThreeDGenerationResponse,
    ThreeDMetrics,
    ThreeDModelItem,
)


def test_3d_schemas_validation():
    """Validates 3D generation Pydantic request and response schemas."""
    req = ThreeDGenerationRequest(image="/images/test.png")
    assert req.model == "dgrauet/hunyuan3d-2.1-mlx"
    assert req.steps == 30
    assert req.guidance_scale == 5.0
    assert req.octree_resolution == 256
    assert req.target_face_count == 40000
    assert req.auto_remove_bg is True
    assert req.pbr_paint is True
    assert req.export_formats == ["glb"]

    metrics = ThreeDMetrics(
        generation_time_s=15.4,
        faces=39500,
        vertices=19800,
        peak_memory_gb=6.2,
        has_pbr_texture=False,
        octree_resolution=256,
        timing_breakdown={"dit_s": 11.2, "vae_decode_s": 3.8},
    )
    assert metrics.faces == 39500
    assert metrics.timing_breakdown["dit_s"] == 11.2

    resp = ThreeDGenerationResponse(
        created=1792000000,
        data=[ThreeDModelItem(glb_url="/models_3d/test.glb")],
        tpm_metrics=metrics,
    )
    assert resp.data[0].glb_url == "/models_3d/test.glb"


def test_3d_preprocessor_centering():
    """Tests image recentering, aspect ratio preservation, and white background compositing."""
    # Create a 200x200 transparent image with a 60x60 red box
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 70, 130, 130], fill=(255, 0, 0, 255))

    processor = ImageProcessorV2(size=518, border_ratio=0.15)
    composited = processor.recenter_and_composite(img)

    assert composited.size == (518, 518)
    assert composited.mode == "RGB"

    # Verify background is pure white
    top_left_pixel = composited.getpixel((5, 5))
    assert top_left_pixel == (255, 255, 255)

    # Test full preprocessing pipeline into MLX array
    tensor, _ = processor.preprocess(img, auto_remove_bg=False, dtype=mx.float32)
    assert tensor.shape == (1, 518, 518, 3)
    assert tensor.dtype == mx.float32


def test_dinov2_mlx_architecture():
    """Validates native MLX DINOv2 vision encoder token extraction."""
    # Small test architecture: image_size=70, patch_size=14 -> 5x5 = 25 patches + 1 CLS = 26 tokens
    model = ImageEncoder(
        image_size=70,
        patch_size=14,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
    )

    dummy_img = mx.zeros((1, 70, 70, 3), dtype=mx.float32)
    features = model(dummy_img, value_range=(-1.0, 1.0))
    mx.eval(features)

    assert features.shape == (1, 26, 64)

    # Unconditional embedding for CFG
    uncond = model.unconditional_embedding(batch_size=2)
    assert uncond.shape == (2, 26, 64)
    assert mx.all(uncond == 0.0).item()


def test_shape_dit_architecture():
    """Validates HunYuanDiTPlain forward pass with U-Net skips and MoE blocks."""
    # Small test DiT: depth=4, last 2 blocks use MoE with 4 experts
    dit = HunYuanDiTPlain(
        input_size=16,
        in_channels=8,
        hidden_size=32,
        context_dim=64,
        depth=4,
        num_heads=4,
        num_moe_layers=2,
        num_experts=4,
        moe_top_k=2,
    )

    latents = mx.zeros((1, 16, 8), dtype=mx.float32)
    t = mx.array([0.5], dtype=mx.float32)
    context = mx.zeros((1, 10, 64), dtype=mx.float32)

    pred = dit(latents, t, {"main": context})
    mx.eval(pred)

    assert pred.shape == (1, 16, 8)


def test_shape_vae_architecture():
    """Validates ShapeVAEDecoder latent decoding and cross-attention SDF query."""
    vae = ShapeVAEDecoder(
        num_latents=16,
        embed_dim=8,
        width=32,
        heads=4,
        num_decoder_layers=2,
        num_freqs=4,
    )

    latents = mx.zeros((1, 16, 8), dtype=mx.float32)
    decoded_features = vae.decode_latents(latents)
    mx.eval(decoded_features)
    assert decoded_features.shape == (1, 16, 32)

    # Query 20 3D spatial points
    queries = mx.zeros((1, 20, 3), dtype=mx.float32)
    sdf = vae.query_sdf(queries, decoded_features)
    mx.eval(sdf)
    assert sdf.shape == (1, 20, 1)


def test_mesh_cleaning_simplification_and_export():
    """Validates mesh post-processing, studio clay material, and GLB export."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a sample icosphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        assert len(sphere.faces) > 0

        # Test mesh cleaning
        cleaned = clean_mesh(sphere)
        assert len(cleaned.faces) == len(sphere.faces)

        # Test simplification
        simplified = simplify_mesh(cleaned, target_faces=50)
        assert len(simplified.faces) <= len(sphere.faces)

        # Test studio clay material
        clay_mesh = apply_studio_clay_material(simplified)
        assert clay_mesh.visual is not None

        # Test export
        base_out = Path(tmp_dir) / "test_output"
        exported = export_mesh_files(clay_mesh, base_out, formats=["glb"])
        assert "glb" in exported

        glb_file = Path(exported["glb"])
        assert glb_file.exists()
        assert glb_file.stat().st_size > 0

        # Reload exported mesh to ensure it is valid
        reloaded = trimesh.load(str(glb_file), force="mesh")
        assert len(reloaded.faces) > 0


def test_mesh_uv_texture_baking_and_pbr_export():
    """Validates Phase 1 xatlas UV parameterization, Metal UV texture baking, and PBR GLB export."""
    from tpm_mlx.diffusion_3d.mesh_extract import bake_mesh_uv_texture

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a cube mesh and a test image with white center emblem
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        ref_img = Image.new("RGBA", (128, 128), (255, 128, 0, 255))
        draw = ImageDraw.Draw(ref_img)
        draw.rectangle([32, 32, 96, 96], fill=(255, 255, 255, 255))  # White emblem

        baked = bake_mesh_uv_texture(box, ref_img, texture_size=512)
        assert baked.visual.kind == "texture"
        assert hasattr(baked.visual.material, "baseColorTexture")
        assert baked.visual.material.baseColorTexture.size == (512, 512)

        out_path = Path(tmp_dir) / "test_pbr_cube"
        exported = export_mesh_files(box, out_path, formats=["glb"], reference_image=ref_img)
        assert "glb" in exported
        reloaded = trimesh.load(exported["glb"], force="mesh")
        assert reloaded.visual.kind == "texture"


def test_stage2_paint_pipeline_initialization():
    """Validates Stage 2 Hunyuan3DPaintPipelineMLX configuration and lazy loading."""
    from tpm_mlx.diffusion_3d.paint.paint_pipeline import (
        Hunyuan3DPaintConfigMLX,
        Hunyuan3DPaintPipelineMLX,
    )

    cfg = Hunyuan3DPaintConfigMLX(max_num_view=6, resolution=512, texture_size=1024)
    assert cfg.max_selected_view_num == 6
    assert cfg.resolution == 512
    assert cfg.texture_size == 1024

    pipeline = Hunyuan3DPaintPipelineMLX(config=cfg, lazy=True)
    assert pipeline.is_loaded is False
    assert pipeline.render is not None
    assert pipeline.view_processor is not None

    pipeline.unload()
    assert pipeline.is_loaded is False


def test_3d_route_endpoint():
    """Tests POST /v1/3d/generations endpoint with mocked MLX3DEngine."""
    from fastapi.testclient import TestClient
    from tpm_mlx.server import create_app
    from tpm_mlx import state

    app = create_app()
    client = TestClient(app)

    mock_metrics = ThreeDMetrics(
        generation_time_s=12.5,
        faces=40000,
        vertices=20100,
        peak_memory_gb=6.4,
        has_pbr_texture=False,
        octree_resolution=256,
    )
    mock_response = ThreeDGenerationResponse(
        created=1792000000,
        data=[ThreeDModelItem(glb_url="/models_3d/test.glb")],
        tpm_metrics=mock_metrics,
    )

    mock_engine = MagicMock()
    mock_engine.generate_mesh.return_value = (mock_response, {"glb": "/static/models_3d/test.glb"})

    with patch.object(state, "get_or_load_3d_engine", return_value=mock_engine):
        response = client.post(
            "/v1/3d/generations",
            json={
                "model": "hunyuan3d-2.1-turbo-mlx",
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                "steps": 8,
                "octree_resolution": 256,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["glb_url"] == "/models_3d/test.glb"
        assert data["tpm_metrics"]["faces"] == 40000


def test_3d_engine_text_prompt_flow():
    """Tests MLX3DEngine.generate_mesh with text prompt chaining to 2D image."""
    from tpm_mlx.engine_3d import MLX3DEngine
    from tpm_mlx.image_engine import ImageResult, ImageGenerationMetrics
    from PIL import Image

    engine = MLX3DEngine()
    mock_pipe = MagicMock()
    mock_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mock_pipe.generate.return_value = (mock_mesh, {"total_s": 1.0})
    engine.pipeline = mock_pipe

    mock_img = Image.new("RGB", (64, 64), color="red")
    mock_res = ImageResult(
        pil_image=mock_img,
        b64_json="",
        png_bytes=b"",
        metrics=ImageGenerationMetrics(),
    )

    req = ThreeDGenerationRequest(
        model="dgrauet/hunyuan3d-2.1-mlx",
        prompt="A cute cartoon robot",
        steps=8,
        octree_resolution=64,
        pbr_paint=False,
        export_formats=["glb"],
    )

    with patch("tpm_mlx.image_engine.MLXImageEngine.generate", return_value=mock_res):
        with tempfile.TemporaryDirectory() as tmp_dir:
            resp, exported = engine.generate_mesh(req, output_dir=tmp_dir)
            assert resp.data[0].glb_url.startswith("/models_3d/")
            assert "glb" in exported
            mock_pipe.generate.assert_called_once()
