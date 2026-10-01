# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
Model catalog inspection and dynamic switching endpoints.
"""

import time
import gc
import asyncio
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
            "status": "ready",
        })

    # If a model is currently in-flight loading
    if state.is_loading and state.loading_model_id and state.loading_model_id not in seen_ids:
        seen_ids.add(state.loading_model_id)
        data.append({
            "id": state.loading_model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tpm-mlx",
            "active": False,
            "active_type": "loading",
            "is_draft": False,
            "is_image": state.is_image_repo(state.loading_model_id),
            "backend": "loading",
            "status": "loading",
        })

    # 2. Retrieve verified cached models on disk
    cached = get_cached_models()
    for item in cached:
        repo_id = item["repo_id"]
        if repo_id in seen_ids:
            continue
        seen_ids.add(repo_id)

        # A draft model in cache is ONLY active if currently loaded base model is an LLM with this draft model
        is_active_draft = bool(not is_curr_img and state.loaded_draft_model_id and repo_id == state.loaded_draft_model_id)
        # An image model in cache is ONLY active if it is the currently loaded base model
        is_active_image = bool(is_curr_img and state.loaded_model_id and repo_id == state.loaded_model_id)

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
            "arch": item.get("arch", ""),
            "status": "ready" if (is_active_image or is_active_draft) else "cached",
        })

    loading_elapsed = round(time.time() - state.loading_start_time, 1) if (state.is_loading and state.loading_start_time) else None

    return {
        "object": "list",
        "data": data,
        "active_model": state.loaded_model_id,
        "active_draft_model": None if is_curr_img else state.loaded_draft_model_id,
        "active_image_model": state.loaded_model_id if is_curr_img else state.loaded_image_model_id,
        "speculation_mode": "none" if is_curr_img else (getattr(state.engine, "speculation_mode", "none") if state.engine else "none"),
        "has_mtp": False if is_curr_img else (getattr(state.engine, "has_mtp", False) if state.engine else False),
        "num_draft_tokens": 0 if is_curr_img else (getattr(state.engine, "num_draft_tokens", 0) if state.engine else 0),
        "backend": "image" if is_curr_img else (getattr(state.engine, "backend", "llm") if state.engine else "llm"),
        "is_loading": state.is_loading,
        "loading_model": state.loading_model_id if state.is_loading else None,
        "loading_elapsed_s": loading_elapsed,
        "status": "loading" if state.is_loading else ("ready" if (state.engine or (is_curr_img and state.image_engine)) else "idle"),
    }


@router.post("/load_model")
@router.post("/models/load")
async def load_model_endpoint(req: LoadModelRequest):
    """
    Endpoint to load/switch models dynamically from the playground, Homer, or API.
    Auto-detects image vs text models and routes accordingly.
    Shields background load tasks against client disconnection timeouts.
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
            "is_loading": False,
        }

    if state.is_image_repo(req.model):
        await state.get_or_load_image_engine(req.model)
        state.loaded_model_id = req.model
        state.loaded_draft_model_id = None
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
            "is_loading": False,
        }

    kv_size = req.max_kv_size or state.default_max_kv_size
    enable_mtp = req.enable_mtp if req.enable_mtp is not None else True
    clean_draft = req.draft_model.strip() if req.draft_model and req.draft_model.strip() else None
    if clean_draft in ("__NONE__", "none", "null", ""):
        clean_draft = None
    elif clean_draft in ("__AUTO__", "auto") or (clean_draft is None and enable_mtp):
        clean_base = req.model.lower().replace("-4bit", "").replace("-8bit", "").replace("-bf16", "").replace("-fp16", "").replace("-it", "")
        clean_base_tag = clean_base.split("/")[-1]
        candidates = []
        for m in get_cached_models():
            if m.get("is_draft"):
                m_id = m["repo_id"]
                m_lower = m_id.lower()
                m_tag = m_lower.split("/")[-1].replace("-4bit", "").replace("-8bit", "").replace("-bf16", "").replace("-fp16", "").replace("-it", "")
                m_without_spec = m_tag.replace("-mtp", "").replace("-assistant", "").replace("_mtp", "").replace("_assistant", "")
                if m_without_spec == clean_base_tag or (clean_base_tag in m_tag and ("mtp" in m_tag or "assistant" in m_tag)):
                    candidates.append(m_id)
        if candidates:
            if "gemma" in req.model.lower():
                bf16_cand = [c for c in candidates if "bf16" in c.lower()]
                clean_draft = bf16_cand[0] if bf16_cand else candidates[0]
            else:
                clean_draft = candidates[0]
            logger.info(f"Auto-paired base model {req.model} with companion draft assistant {clean_draft}")
        else:
            clean_draft = None

    # De-duplicate: attach to existing task if same model is currently loading
    if state.is_loading and state.loading_model_id == req.model and state.loading_task and not state.loading_task.done():
        load_task = state.loading_task
    else:
        load_coro = state.load_engine(
            model_id=req.model,
            max_kv_size=kv_size,
            draft_model=clean_draft,
            enable_mtp=enable_mtp,
            num_draft_tokens=req.num_draft_tokens,
        )
        load_task = asyncio.create_task(load_coro)
        state.loading_task = load_task

    if req.async_load:
        return {
            "status": "loading",
            "message": f"Model {req.model} is loading in the background.",
            "model": req.model,
            "draft_model": clean_draft,
            "is_loading": True,
        }

    try:
        # Shield the load_task so client disconnect/timeout (e.g. 30s mobile socket timeout)
        # does not cancel background model allocation and assignment to state.engine
        await asyncio.shield(load_task)
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
            "is_loading": False,
        }
    except asyncio.CancelledError:
        logger.warning(f"Client disconnected/timed out while loading {req.model}; task will continue in background.")
        raise HTTPException(
            status_code=499,
            detail=f"Client disconnected; model {req.model} is continuing to load in background."
        )
    except Exception as e:
        logger.error(f"Error loading model {req.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load_image_model")
async def load_image_model_endpoint(req: LoadImageModelRequest):
    """Dynamically loads or switches the active image generation model."""
    await state.get_or_load_image_engine(req.model)
    return JSONResponse(content={"status": "ok", "loaded_image_model": req.model})
