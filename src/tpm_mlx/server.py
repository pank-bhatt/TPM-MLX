# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
FastAPI Server for TPM-MLX.
Provides OpenAI-compatible endpoints for LLM chat completions, multimodal vision,
flow matching image generation, and modern browser playground serving.
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tpm_mlx import state
from tpm_mlx import director
from tpm_mlx import vlm
from tpm_mlx.routes import (
    views_router,
    models_router,
    chat_router,
    images_router,
    embeddings_router,
    threed_router,
)
from tpm_mlx.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    LoadModelRequest,
    LoadImageModelRequest,
    ImageGenerationApiRequest,
    ImageEditApiRequest,
)
from tpm_mlx.utils import get_logger, install_safe_streams

install_safe_streams()
logger = get_logger("server")

# Backwards compatibility re-exports for benchmarks and external integrations
_get_or_load_image_engine = state.get_or_load_image_engine
_load_engine = state.load_engine
_distill_visual_prompt = director.distill_visual_prompt
_extract_text_and_images = vlm.extract_text_and_images


async def _idle_eviction_watchdog():
    """Periodically checks loaded models and safely evicts idle engines from Unified Memory.

    Guarantees:
    - Never evicts while state.busy_counts[modality] > 0
    - For 3D: verifies no async tasks in 'processing' or 'queued' state
    - Checks (now - state.last_completed[modality]) > timeout
    - Executes eviction strictly on state.mlx_executor inside corresponding loading lock
    """
    logger.info("Safe Idle Eviction Watchdog initialized.")
    while True:
        try:
            await asyncio.sleep(15)
            now = time.time()

            # 1. 3D Engine Idle Check
            timeout_3d = state.IDLE_TIMEOUT_3D
            if (
                timeout_3d > 0
                and state.three_d_engine is not None
                and state.busy_counts.get("3d", 0) == 0
                and (now - state.last_completed.get("3d", 0.0)) > timeout_3d
            ):
                from tpm_mlx.routes.threed import TASKS
                active_tasks = any(t.status in ("processing", "queued") for t in TASKS.values())
                if not active_tasks and not state.three_d_loading_lock.locked():
                    async with state.three_d_loading_lock:
                        if state.three_d_engine is not None and state.busy_counts.get("3d", 0) == 0:
                            idle_sec = int(now - state.last_completed.get("3d", 0.0))
                            logger.info(f"Safe Idle Eviction: 3D engine idle for {idle_sec}s (threshold {timeout_3d}s). Reclaiming Unified Memory...")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(state.mlx_executor, state.unload_3d_engine)

            # 2. Image Diffusion Engine Idle Check
            timeout_img = state.IDLE_TIMEOUT_IMAGE
            if (
                timeout_img > 0
                and state.image_engine is not None
                and state.busy_counts.get("image", 0) == 0
                and (now - state.last_completed.get("image", 0.0)) > timeout_img
            ):
                if not state.image_loading_lock.locked():
                    async with state.image_loading_lock:
                        if state.image_engine is not None and state.busy_counts.get("image", 0) == 0:
                            idle_sec = int(now - state.last_completed.get("image", 0.0))
                            logger.info(f"Safe Idle Eviction: Image engine idle for {idle_sec}s (threshold {timeout_img}s). Reclaiming Unified Memory...")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(state.mlx_executor, state.unload_image_engine)

            # 3. Text LLM Engine Idle Check
            timeout_text = state.IDLE_TIMEOUT_TEXT
            if (
                timeout_text > 0
                and state.engine is not None
                and state.busy_counts.get("text", 0) == 0
                and (now - state.last_completed.get("text", 0.0)) > timeout_text
                and not state.is_loading
            ):
                if not state.model_loading_lock.locked():
                    async with state.model_loading_lock:
                        if state.engine is not None and state.busy_counts.get("text", 0) == 0:
                            idle_sec = int(now - state.last_completed.get("text", 0.0))
                            logger.info(f"Safe Idle Eviction: Text engine idle for {idle_sec}s (threshold {timeout_text}s). Reclaiming Unified Memory...")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(state.mlx_executor, state.unload_text_engine)

            # 4. Embedding Engine Idle Check
            timeout_emb = state.IDLE_TIMEOUT_EMBEDDING
            if (
                timeout_emb > 0
                and state.embedding_engine is not None
                and state.busy_counts.get("embedding", 0) == 0
                and (now - state.last_completed.get("embedding", 0.0)) > timeout_emb
            ):
                if not state.embedding_loading_lock.locked():
                    async with state.embedding_loading_lock:
                        if state.embedding_engine is not None and state.busy_counts.get("embedding", 0) == 0:
                            idle_sec = int(now - state.last_completed.get("embedding", 0.0))
                            logger.info(f"Safe Idle Eviction: Embedding engine idle for {idle_sec}s (threshold {timeout_emb}s). Reclaiming Unified Memory...")
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(state.mlx_executor, state.unload_embedding_engine)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Error in idle eviction watchdog tick: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan context manager.
    Handles startup model initialization and graceful shutdown resource cleanup.
    """
    # 0. Configure Metal cache cap to prevent unbounded buffer growth
    from tpm_mlx.image_engine import configure_metal_memory, reclaim_metal_memory
    configure_metal_memory()

    # 1. Startup: Load default models configured via environment variables
    default_model = os.environ.get("TPM_DEFAULT_MODEL")
    draft_model = os.environ.get("TPM_DEFAULT_DRAFT_MODEL")
    kv_size = int(os.environ.get("TPM_MAX_KV_SIZE", str(state.default_max_kv_size)))
    enable_mtp = os.environ.get("TPM_ENABLE_MTP", "True").lower() == "true"
    num_draft = int(os.environ.get("TPM_NUM_DRAFT_TOKENS")) if os.environ.get("TPM_NUM_DRAFT_TOKENS") else None

    if default_model and default_model.strip().lower() not in ("none", "null", "false", ""):
        try:
            await state.load_engine(
                model_id=default_model,
                max_kv_size=kv_size,
                draft_model=draft_model,
                enable_mtp=enable_mtp,
                num_draft_tokens=num_draft,
            )
        except Exception as e:
            logger.error(f"Failed to load default model {default_model} on startup: {e}")

    default_image_model = os.environ.get("TPM_DEFAULT_IMAGE_MODEL")
    if default_image_model:
        try:
            await state.get_or_load_image_engine(default_image_model)
        except Exception as e:
            logger.error(f"Failed to load default image model {default_image_model} on startup: {e}")

    # Start the safe idle eviction watchdog background task
    watchdog_task = asyncio.create_task(_idle_eviction_watchdog())

    try:
        yield
    finally:
        # Cancel watchdog task
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

        # 2. Shutdown: Clean up engines and flush Metal Unified Memory caches
        if state.engine is not None:
            try:
                del state.engine
            except Exception:
                pass
        if getattr(state, "image_engine", None) is not None:
            try:
                state.image_engine.unload()
            except Exception:
                pass
        if getattr(state, "embedding_engine", None) is not None:
            try:
                state.unload_embedding_engine()
            except Exception:
                pass
        if getattr(state, "three_d_engine", None) is not None:
            try:
                state.unload_3d_engine()
            except Exception:
                pass
        reclaim_metal_memory()
        logger.info("TPM-MLX server shutdown complete.")


def create_app() -> FastAPI:
    """Factory function to instantiate and configure the FastAPI application."""
    app_instance = FastAPI(
        title="TPM-MLX Server",
        description="Optimized Apple Silicon Inference Engine API Server",
        version="0.5.0",
        lifespan=lifespan,
    )

    # Mount static assets, generated image outputs, and 3D models
    app_instance.mount("/static", StaticFiles(directory=str(state.static_dir)), name="static")
    app_instance.mount("/images", StaticFiles(directory=str(state.images_dir)), name="images")
    app_instance.mount("/models_3d", StaticFiles(directory=str(state.models_3d_dir)), name="models_3d")

    # Enable CORS for integrations (Continue, VS Code, Browser Playground)
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register modular route layers
    app_instance.include_router(views_router)
    app_instance.include_router(models_router)
    app_instance.include_router(chat_router)
    app_instance.include_router(images_router)
    app_instance.include_router(embeddings_router)
    app_instance.include_router(threed_router)

    return app_instance


# Top-level ASGI application instance
app = create_app()


def __getattr__(name: str) -> Any:
    """Dynamic attribute proxy to state module for backward compatibility."""
    if hasattr(state, name):
        return getattr(state, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
