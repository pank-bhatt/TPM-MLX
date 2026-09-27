# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
OpenAI-compatible Image generation, editing, and asset upload endpoints.
Supports FLUX.2 Klein, Z-Image Turbo, Bonsai, and native diffusion architectures on MLX.
"""

import time
import random
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from tpm_mlx import state
from tpm_mlx.schemas.images import ImageGenerationApiRequest, ImageEditApiRequest
from tpm_mlx.director import distill_visual_prompt

logger = logging.getLogger("tpm-mlx.routes.images")
router = APIRouter(prefix="/v1", tags=["images"])

# Re-export for compatibility with tests and callers
_get_or_load_image_engine = state.get_or_load_image_engine


@router.post("/images/generations")
async def generate_images_endpoint(req: ImageGenerationApiRequest):
    """
    OpenAI-compatible image generation endpoint.
    Supports resolution, steps, seed, response_format (url / b64_json),
    and intelligent context distillation via the active text LLM engine.
    """
    import tpm_mlx.server
    img_eng = await tpm_mlx.server._get_or_load_image_engine(req.model)

    effective_prompt = req.prompt
    revised_prompt: Optional[str] = None

    # Context Distillation: Strictly bypass if auto_expand is False. Only distill if True or (None with context/length)
    should_distill = False
    if req.auto_expand is True:
        should_distill = True
    elif req.auto_expand is None and (bool(req.context) or len(req.prompt) > 300):
        should_distill = True

    if should_distill and state.engine is not None:
        revised = await distill_visual_prompt(req.prompt, req.context, is_edit=False)
        if revised and revised != req.prompt and len(revised) >= 10:
            effective_prompt = revised
            revised_prompt = revised
            logger.info(f"Distilled visual prompt: '{effective_prompt[:70]}...'")

    from tpm_mlx.image_engine import MLXImageEngine
    detected_size = MLXImageEngine.extract_resolution_from_text(req.prompt)
    effective_size = detected_size or req.size or "1024x1024"

    # Generate image on dedicated mlx_executor thread
    loop = asyncio.get_running_loop()

    def run_gen():
        filename = f"gen_{int(time.time())}_{random.randint(1000, 9999)}.png"
        filepath = state.images_dir / filename
        res = img_eng.generate(
            prompt=effective_prompt,
            size=effective_size,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            output_path=filepath,
            revised_prompt=revised_prompt or effective_prompt,
            image_paths=req.image,
        )
        return res, filename

    try:
        result, fname = await loop.run_in_executor(state.mlx_executor, run_gen)
    except asyncio.CancelledError:
        raise HTTPException(status_code=503, detail="Image generation cancelled.")
    except Exception as e:
        logger.error(f"Image generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    url = f"/images/{fname}"
    data_item: Dict[str, Any] = {
        "url": url,
        "revised_prompt": result.revised_prompt,
    }
    if req.response_format == "b64_json":
        data_item["b64_json"] = result.b64_json

    return JSONResponse(content={
        "created": int(time.time()),
        "data": [data_item],
        "tpm_metrics": {
            "generation_time_s": result.metrics.generation_time_s,
            "steps": result.metrics.steps,
            "step_time_s": result.metrics.step_time_s,
            "peak_memory_gb": result.metrics.peak_memory_gb,
            "width": result.metrics.width,
            "height": result.metrics.height,
            "seed": result.metrics.seed,
            "model": result.metrics.model,
            "supports_editing": result.metrics.supports_editing,
        }
    })


@router.post("/upload_image")
async def upload_image_endpoint(file: UploadFile = File(...)):
    """Uploads a local user image for Image-to-Image editing."""
    uploads_dir = state.images_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "image.png").suffix or ".png"
    filename = f"upload_{int(time.time())}_{random.randint(1000, 9999)}{ext}"
    dest_path = uploads_dir / filename

    contents = await file.read()
    dest_path.write_bytes(contents)

    return JSONResponse(content={
        "status": "success",
        "filename": filename,
        "url": f"/images/uploads/{filename}",
        "path": str(dest_path),
    })


@router.post("/images/edits")
async def edit_images_endpoint(req: ImageEditApiRequest):
    """
    OpenAI-compatible image editing endpoint.
    Modifies an input reference image according to a natural language prompt.
    """
    import tpm_mlx.server
    img_eng = await tpm_mlx.server._get_or_load_image_engine(req.model)

    if not img_eng.supports_editing:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{img_eng.model_id}' does not support instruction-based image editing. Please select 'mlx-community/FLUX.2-klein-9B'."
        )

    effective_prompt = req.prompt
    revised_prompt: Optional[str] = None

    # Context Distillation: Strictly bypass if auto_expand is False. Only distill if True or (None with context/length)
    should_distill = False
    if req.auto_expand is True:
        should_distill = True
    elif req.auto_expand is None and (bool(req.context) or len(req.prompt) > 300):
        should_distill = True

    if should_distill and state.engine is not None:
        revised = await distill_visual_prompt(req.prompt, req.context, is_edit=True)
        if revised and revised != req.prompt and len(revised) >= 10:
            effective_prompt = revised
            revised_prompt = revised
            logger.info(f"Distilled visual edit prompt: '{effective_prompt[:70]}...'")

    from tpm_mlx.image_engine import MLXImageEngine
    detected_size = MLXImageEngine.extract_resolution_from_text(req.prompt)
    effective_size = detected_size or req.size

    loop = asyncio.get_running_loop()

    def run_edit():
        filename = f"edit_{int(time.time())}_{random.randint(1000, 9999)}.png"
        filepath = state.images_dir / filename
        res = img_eng.edit(
            image_paths=req.image,
            prompt=effective_prompt,
            size=effective_size,
            steps=req.steps,
            guidance=req.guidance,
            seed=req.seed,
            output_path=filepath,
            revised_prompt=revised_prompt or effective_prompt,
        )
        return res, filename

    try:
        result, fname = await loop.run_in_executor(state.mlx_executor, run_edit)
    except asyncio.CancelledError:
        raise HTTPException(status_code=503, detail="Image edit cancelled.")
    except Exception as e:
        logger.error(f"Image edit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")

    url = f"/images/{fname}"
    data_item: Dict[str, Any] = {
        "url": url,
        "revised_prompt": result.revised_prompt,
    }
    if req.response_format == "b64_json":
        data_item["b64_json"] = result.b64_json

    return JSONResponse(content={
        "created": int(time.time()),
        "data": [data_item],
        "tpm_metrics": {
            "generation_time_s": result.metrics.generation_time_s,
            "steps": result.metrics.steps,
            "step_time_s": result.metrics.step_time_s,
            "peak_memory_gb": result.metrics.peak_memory_gb,
            "width": result.metrics.width,
            "height": result.metrics.height,
            "seed": result.metrics.seed,
            "model": result.metrics.model,
            "supports_editing": result.metrics.supports_editing,
        }
    })
