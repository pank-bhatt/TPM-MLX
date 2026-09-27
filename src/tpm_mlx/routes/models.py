# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
Model catalog inspection and dynamic switching endpoints.
"""

import time
import gc
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import mlx.core as mx

from tpm_mlx import state
from tpm_mlx.schemas.models import LoadModelRequest, LoadImageModelRequest
from tpm_mlx.utils import get_cached_models

logger = logging.getLogger("tpm-mlx.routes.models")
router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models():
    """
    Returns list of loaded models and Hugging Face cached models.
    """
    data = []
    seen_ids = set()

    # 1. Add currently loaded text/vision LLM or image model if available
    is_curr_img = state.is_image_repo(state.loaded_model_id) if state.loaded_model_id else False
    if state.loaded_model_id:
        seen_ids.add(state.loaded_model_id)
        data.append({
            "id": state.loaded_model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tpm-mlx",
            "active": True,
            "active_type": "image" if is_curr_img else "llm",
            "is_draft": False,
            "is_image": is_curr_img,
            "backend": "image" if is_curr_img else getattr(state.engine, "backend", "llm"),
            "max_kv_size": getattr(state.engine, "max_kv_size", state.default_max_kv_size),
            "speculation_mode": "none" if is_curr_img else getattr(state.engine, "speculation_mode", "none"),
            "has_mtp": False if is_curr_img else getattr(state.engine, "has_mtp", False),
            "num_draft_tokens": 0 if is_curr_img else getattr(state.engine, "num_draft_tokens", 0),
            "draft_model": None if is_curr_img else state.loaded_draft_model_id,
        })

    # 2. Retrieve verified cached models on disk
    cached = get_cached_models()
    for item in cached:
        repo_id = item["repo_id"]
        if repo_id in seen_ids:
            continue
        seen_ids.add(repo_id)

        is_active_image = bool(state.loaded_image_model_id and repo_id == state.loaded_image_model_id)
        is_active_draft = bool(state.loaded_draft_model_id and repo_id == state.loaded_draft_model_id)

        data.append({
            "id": repo_id,
            "object": "model",
            "created": int(item["last_modified"]),
            "owned_by": "huggingface",
            "active": is_active_image or is_active_draft,
            "active_type": "image" if is_active_image else ("draft" if is_active_draft else None),
            "is_draft": item.get("is_draft", False),
            "is_image": item.get("is_image", False),
            "size_bytes": item.get("size_on_disk", 0),
            "arch": item.get("arch", "")
        })

    return {
        "object": "list",
        "data": data,
        "active_model": state.loaded_model_id,
        "active_draft_model": None if is_curr_img else state.loaded_draft_model_id,
        "active_image_model": state.loaded_image_model_id,
        "speculation_mode": "none" if is_curr_img else (getattr(state.engine, "speculation_mode", "none") if state.engine else "none"),
        "has_mtp": False if is_curr_img else (getattr(state.engine, "has_mtp", False) if state.engine else False),
        "num_draft_tokens": 0 if is_curr_img else (getattr(state.engine, "num_draft_tokens", 0) if state.engine else 0),
        "backend": "image" if is_curr_img else (getattr(state.engine, "backend", "llm") if state.engine else "llm"),
    }


@router.post("/load_model")
async def load_model_endpoint(req: LoadModelRequest):
    """
    Endpoint to load/switch models dynamically from the playground or API.
    Auto-detects image vs text models and routes accordingly.
    """
    if req.model.strip().lower() in ("none", "null", "unload"):
        old_engine = state.engine
        state.engine = None
        state.loaded_model_id = None
        state.loaded_draft_model_id = None
        if old_engine is not None:
            try:
                del old_engine.model
                del old_engine.processor
                del old_engine.drafter
            except Exception:
                pass
            del old_engine
        gc.collect()
        mx.clear_cache()
        return {
            "status": "success",
            "message": "Unloaded text engine from Unified Memory",
            "model": None,
            "draft_model": None,
            "backend": "none",
            "speculation_mode": "none",
            "has_mtp": False,
            "num_draft_tokens": 0,
            "is_image": False,
        }

    if state.is_image_repo(req.model):
        await state.get_or_load_image_engine(req.model)
        state.loaded_model_id = req.model
        return {
            "status": "success",
            "message": f"Successfully loaded image model {req.model}",
            "model": req.model,
            "draft_model": None,
            "backend": "image",
            "speculation_mode": "none",
            "has_mtp": False,
            "num_draft_tokens": 0,
            "is_image": True,
        }

    kv_size = req.max_kv_size or state.default_max_kv_size
    enable_mtp = req.enable_mtp if req.enable_mtp is not None else True
    try:
        await state.load_engine(
            model_id=req.model,
            max_kv_size=kv_size,
            draft_model=req.draft_model,
            enable_mtp=enable_mtp,
            num_draft_tokens=req.num_draft_tokens,
        )
        return {
            "status": "success",
            "message": f"Successfully loaded model {req.model}",
            "model": req.model,
            "draft_model": state.loaded_draft_model_id,
            "backend": state.engine.backend if state.engine else "llm",
            "speculation_mode": state.engine.speculation_mode if state.engine else "none",
            "has_mtp": state.engine.has_mtp if state.engine else False,
            "num_draft_tokens": state.engine.num_draft_tokens if state.engine else 0,
            "is_image": False,
        }
    except Exception as e:
        logger.error(f"Error loading model {req.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load_image_model")
async def load_image_model_endpoint(req: LoadImageModelRequest):
    """Dynamically loads or switches the active image generation model."""
    await state.get_or_load_image_engine(req.model)
    return JSONResponse(content={"status": "ok", "loaded_image_model": req.model})
