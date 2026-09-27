# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
FastAPI Server for TPM-MLX.
Provides OpenAI-compatible endpoints for LLM chat completions, multimodal vision,
flow matching image generation, and modern browser playground serving.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tpm_mlx import state
from tpm_mlx import director
from tpm_mlx import vlm
from tpm_mlx.routes import views_router, models_router, chat_router, images_router
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan context manager.
    Handles startup model initialization and graceful shutdown resource cleanup.
    """
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

    yield

    # 2. Shutdown: Clean up engines and flush Metal Unified Memory caches
    if state.engine is not None:
        try:
            del state.engine
        except Exception:
            pass
    if state.image_engine is not None:
        try:
            state.image_engine.unload()
        except Exception:
            pass
    logger.info("TPM-MLX server shutdown complete.")


def create_app() -> FastAPI:
    """Factory function to instantiate and configure the FastAPI application."""
    app_instance = FastAPI(
        title="TPM-MLX Server",
        description="Optimized Apple Silicon Inference Engine API Server",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Mount static assets and generated image outputs
    app_instance.mount("/static", StaticFiles(directory=str(state.static_dir)), name="static")
    app_instance.mount("/images", StaticFiles(directory=str(state.images_dir)), name="images")

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

    return app_instance


# Top-level ASGI application instance
app = create_app()


def __getattr__(name: str) -> Any:
    """Dynamic attribute proxy to state module for backward compatibility."""
    if hasattr(state, name):
        return getattr(state, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
