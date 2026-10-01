# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
Centralized application state and model lifecycle management for TPM-MLX.
Manages global MLX thread affinity, model loading locks, KV caches,
and unified memory cache eviction for both LLM and Diffusion image engines.
"""

import os
import gc
import time
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

import mlx.core as mx

from tpm_mlx.engine import MLXEngine
from tpm_mlx.utils import get_logger

logger = get_logger("state")

# MLX requires GPU stream affinity; all MLX ops run on a dedicated single-thread worker.
mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_thread")

# Global text/vision LLM engine state
engine: Optional[MLXEngine] = None
loaded_model_id: Optional[str] = None
loaded_draft_model_id: Optional[str] = None
model_loading_lock = asyncio.Lock()

# Model loading lifecycle telemetry and in-flight tracking
is_loading: bool = False
loading_model_id: Optional[str] = None
loading_start_time: Optional[float] = None
loading_task: Optional[asyncio.Task] = None
loading_error: Optional[str] = None

# Global image diffusion engine state (lazy-loaded on demand)
image_engine: Optional[Any] = None
loaded_image_model_id: Optional[str] = None
image_loading_lock = asyncio.Lock()

# Directory paths for static frontend assets and generated images
static_dir = Path(__file__).parent / "static"
images_dir = static_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Default pre-allocated KV cache capacity
default_max_kv_size = 4096


def is_image_repo(repo_id: Optional[str]) -> bool:
    """Checks whether a Hugging Face repository ID corresponds to an image diffusion model."""
    if not repo_id:
        return False
    lower = repo_id.lower()
    return (
        any(k in lower for k in ("flux", "diffusion", "klein", "sdxl", "stable-diffusion", "z_image", "z-image", "zimage"))
        or ("bonsai" in lower and "image" in lower)
    )


async def get_or_load_image_engine(model_id: Optional[str] = None) -> Any:
    """
    Dynamically retrieves or initializes the MLX Image Engine.
    Cleans up old weights and flushes Metal cache when switching diffusion models.
    """
    global image_engine, loaded_image_model_id, mlx_executor
    from tpm_mlx.image_engine import MLXImageEngine

    target_model = model_id or loaded_image_model_id or MLXImageEngine.DEFAULT_MODEL

    if image_engine is not None and loaded_image_model_id == target_model:
        return image_engine

    async with image_loading_lock:
        if image_engine is not None and loaded_image_model_id == target_model:
            return image_engine

        logger.info(f"Loading image engine with model: {target_model}...")

        # Clean up existing image engine if changing models to free unified memory
        old_img = image_engine
        image_engine = None
        if old_img is not None and loaded_image_model_id != target_model:
            try:
                old_img.unload()
            except Exception:
                pass
            del old_img
            gc.collect()
            try:
                mx.clear_cache()
                if hasattr(mx, "metal"):
                    mx.metal.clear_cache()
            except Exception:
                pass

        def init_img():
            try:
                return MLXImageEngine(model_path_or_id=target_model, lazy=False)
            except BrokenPipeError:
                logger.warning("Caught BrokenPipeError during image engine load; retrying once...")
                time.sleep(0.5)
                return MLXImageEngine(model_path_or_id=target_model, lazy=False)

        loop = asyncio.get_running_loop()
        new_img_engine = await loop.run_in_executor(mlx_executor, init_img)
        image_engine = new_img_engine
        loaded_image_model_id = target_model
        return image_engine


async def load_engine(
    model_id: str,
    max_kv_size: int,
    draft_model: Optional[str] = None,
    enable_mtp: bool = True,
    num_draft_tokens: Optional[int] = None,
) -> MLXEngine:
    """
    Dynamically loads an LLM/VLM model into unified memory with static KV pre-allocation
    and speculative decoding (MTP head or companion draft assistant).
    """
    global engine, loaded_model_id, loaded_draft_model_id, mlx_executor
    global is_loading, loading_model_id, loading_start_time, loading_task, loading_error

    clean_draft = draft_model if draft_model and draft_model.strip() else None

    # 1. Immediate return if model is already loaded and active in memory
    if (
        engine is not None
        and loaded_model_id == model_id
        and loaded_draft_model_id == clean_draft
        and getattr(engine, "max_kv_size", default_max_kv_size) == max_kv_size
    ):
        logger.info(f"Model {model_id} already loaded and ready in Unified Memory.")
        return engine

    # 2. De-duplicate if load for this exact model is currently active in background
    if is_loading and loading_model_id == model_id and loading_task is not None and not loading_task.done():
        logger.info(f"Model {model_id} is already loading in background; attaching to active task...")
        return await asyncio.shield(loading_task)

    async with model_loading_lock:
        # Re-check inside lock
        if (
            engine is not None
            and loaded_model_id == model_id
            and loaded_draft_model_id == clean_draft
            and getattr(engine, "max_kv_size", default_max_kv_size) == max_kv_size
        ):
            return engine

        is_loading = True
        loading_model_id = model_id
        loading_start_time = time.time()
        loading_error = None

        logger.info(f"Loading model: {model_id} (KV Cache Size: {max_kv_size}, Draft: {clean_draft}, MTP: {enable_mtp})...")
        start_time = time.perf_counter()

        # Clean up existing engine and release Metal GPU buffers before loading new model
        old_engine = engine
        engine = None
        loaded_model_id = None
        loaded_draft_model_id = None

        if old_engine is not None:
            try:
                del old_engine.model
                del old_engine.processor
                del old_engine.drafter
                del old_engine.tokenizer
            except Exception:
                pass
            del old_engine

        def init_engine():
            gc.collect()
            mx.clear_cache()
            if hasattr(mx, "metal"):
                mx.metal.clear_cache()

            try:
                eng = MLXEngine(
                    model_path_or_id=model_id,
                    max_kv_size=max_kv_size,
                    draft_model_path_or_id=clean_draft,
                    enable_mtp=enable_mtp,
                    num_draft_tokens=num_draft_tokens,
                )
            except BrokenPipeError:
                logger.warning("Caught BrokenPipeError during MLX engine load; retrying once...")
                time.sleep(0.5)
                eng = MLXEngine(
                    model_path_or_id=model_id,
                    max_kv_size=max_kv_size,
                    draft_model_path_or_id=clean_draft,
                    enable_mtp=enable_mtp,
                    num_draft_tokens=num_draft_tokens,
                )

            gc.collect()
            mx.clear_cache()
            if hasattr(mx, "metal"):
                mx.metal.clear_cache()

            return eng

        try:
            loop = asyncio.get_running_loop()
            new_engine = await loop.run_in_executor(mlx_executor, init_engine)

            engine = new_engine
            loaded_model_id = model_id
            loaded_draft_model_id = clean_draft
            duration = time.perf_counter() - start_time
            logger.info(f"Successfully loaded {model_id} [Speculation: {engine.speculation_mode.upper()}] in {duration:.2f}s")
            return engine
        except Exception as e:
            loading_error = str(e)
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            engine = None
            loaded_model_id = None
            loaded_draft_model_id = None
            raise
        finally:
            is_loading = False
            loading_model_id = None
            loading_task = None


# Backwards compatibility aliases
_get_or_load_image_engine = get_or_load_image_engine
_load_engine = load_engine
