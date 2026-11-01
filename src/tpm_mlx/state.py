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
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict

import mlx.core as mx

from tpm_mlx.engine import MLXEngine
from tpm_mlx.utils import get_active_memory_gb, get_cached_models, get_logger, reclaim_metal_memory

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

# Global embedding engine state (EmbeddingGemma 2 4-bit)
embedding_engine: Optional[Any] = None
loaded_embedding_model_id: Optional[str] = None
embedding_loading_lock = asyncio.Lock()
DEFAULT_EMBEDDING_MODEL = os.environ.get("TPM_DEFAULT_EMBEDDING_MODEL", "mlx-community/embeddinggemma-2-4bit")

# Global 3D generation engine state (Hunyuan3D 2.1)
three_d_engine: Optional[Any] = None
loaded_3d_model_id: Optional[str] = None
three_d_loading_lock = asyncio.Lock()
DEFAULT_3D_MODEL = os.environ.get("TPM_DEFAULT_3D_MODEL", "dgrauet/hunyuan3d-2.1-mlx")

# Directory paths for static frontend assets, generated images, and 3D models
static_dir = Path(__file__).parent / "static"
images_dir = static_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
models_3d_dir = static_dir / "models_3d"
models_3d_dir.mkdir(parents=True, exist_ok=True)

# Default pre-allocated KV cache capacity
default_max_kv_size = 4096

# Idle eviction and active request tracking
# Default idle timeouts in seconds (0 to disable eviction):
# 3D: 300s (5 min) - large ~7 GB shape pipeline
# Image: 300s (5 min) - large ~8.5 GB diffusion pipeline
# Text: 900s (15 min) - primary conversational LLM
# Embedding: 600s (10 min) - embedding engine
DEFAULT_IDLE_TIMEOUT = int(os.environ.get("TPM_IDLE_TIMEOUT", "300"))
IDLE_TIMEOUT_3D = int(os.environ.get("TPM_IDLE_TIMEOUT_3D", str(DEFAULT_IDLE_TIMEOUT)))
IDLE_TIMEOUT_IMAGE = int(os.environ.get("TPM_IDLE_TIMEOUT_IMAGE", str(DEFAULT_IDLE_TIMEOUT)))
IDLE_TIMEOUT_TEXT = int(os.environ.get("TPM_IDLE_TIMEOUT_TEXT", str(os.environ.get("TPM_IDLE_TIMEOUT", "900"))))
IDLE_TIMEOUT_EMBEDDING = int(os.environ.get("TPM_IDLE_TIMEOUT_EMBEDDING", "600"))

busy_counts: Dict[str, int] = {
    "text": 0,
    "image": 0,
    "3d": 0,
    "embedding": 0,
}

last_completed: Dict[str, float] = {
    "text": time.time(),
    "image": 0.0,
    "3d": 0.0,
    "embedding": 0.0,
}


@contextmanager
def active_inference(modality: str):
    """Context manager to guarantee an engine is never evicted while actively executing inference.

    Increments in-flight busy count on entry; decrements and marks last_completed on exit.
    """
    busy_counts[modality] = busy_counts.get(modality, 0) + 1
    try:
        yield
    finally:
        busy_counts[modality] = max(0, busy_counts.get(modality, 1) - 1)
        last_completed[modality] = time.time()


def unload_all_engines() -> dict:
    """Evicts all active idle engines (Text, Image, 3D, Embedding) and flushes Metal Unified Memory."""
    unloaded = []
    if engine is not None and busy_counts.get("text", 0) == 0:
        unload_text_engine()
        unloaded.append("text")
    if image_engine is not None and busy_counts.get("image", 0) == 0:
        unload_image_engine()
        unloaded.append("image")
    if three_d_engine is not None and busy_counts.get("3d", 0) == 0:
        unload_3d_engine()
        unloaded.append("3d")
    if embedding_engine is not None and busy_counts.get("embedding", 0) == 0:
        unload_embedding_engine()
        unloaded.append("embedding")
    reclaim_metal_memory()
    return {
        "unloaded": unloaded,
        "active_memory_gb": round(get_active_memory_gb(), 2),
    }


def is_3d_repo(repo_id: Optional[str]) -> bool:
    """Checks whether a Hugging Face repository ID corresponds to a 3D geometry model."""
    if not repo_id:
        return False
    lower = repo_id.lower()
    return any(k in lower for k in ("3d", "hunyuan3d", "hy3d", "shap-e", "triposr", "point-e"))


def is_image_repo(repo_id: Optional[str]) -> bool:
    """Checks whether a Hugging Face repository ID corresponds to an image diffusion model."""
    if not repo_id or is_3d_repo(repo_id):
        return False
    lower = repo_id.lower()
    return (
        any(k in lower for k in ("flux", "diffusion", "klein", "sdxl", "stable-diffusion", "z_image", "z-image", "zimage", "qwen-image", "qwen_image", "studio-image"))
        or ("bonsai" in lower and "image" in lower)
        or ("qwen" in lower and "image" in lower)
    )


def unload_text_engine() -> None:
    """
    Unloads active LLM/VLM text engine and flushes Unified Memory and Metal GPU caches.
    """
    global engine, loaded_model_id, loaded_draft_model_id
    if engine is not None:
        logger.info(f"Mutual Unload: Evicting text engine '{loaded_model_id}' from Unified Memory...")
        old_engine = engine
        engine = None
        loaded_model_id = None
        loaded_draft_model_id = None
        try:
            del old_engine.model
            del old_engine.processor
            del old_engine.drafter
            del old_engine.tokenizer
        except Exception:
            pass
        del old_engine
        reclaim_metal_memory()
        logger.info(f"Text engine evicted. Active Unified Memory: {get_active_memory_gb():.2f} GB")


def unload_image_engine() -> None:
    """
    Unloads active image diffusion engine and flushes Unified Memory and Metal GPU caches.
    """
    global image_engine, loaded_image_model_id
    if image_engine is not None:
        logger.info(f"Mutual Unload: Evicting image engine '{loaded_image_model_id}' from Unified Memory...")
        old_img = image_engine
        image_engine = None
        loaded_image_model_id = None
        try:
            old_img.unload()
        except Exception:
            pass
        del old_img
        reclaim_metal_memory()
def unload_3d_engine() -> None:
    """
    Unloads active 3D generation engine and flushes Unified Memory and Metal GPU caches.
    """
    global three_d_engine, loaded_3d_model_id
    if three_d_engine is not None:
        logger.info(f"Mutual Unload: Evicting 3D engine '{loaded_3d_model_id}' from Unified Memory...")
        old_3d = three_d_engine
        three_d_engine = None
        loaded_3d_model_id = None
        try:
            old_3d.unload()
        except Exception:
            pass
        del old_3d
        reclaim_metal_memory()
        logger.info(f"3D engine evicted. Active Unified Memory: {get_active_memory_gb():.2f} GB")


def evict_peer_instance(port: int = 2506) -> None:
    """Non-blocking cross-daemon eviction request to reclaim external heavy engine VRAM."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/unload",
            data=b'{"modality": "all"}',
            headers={"Content-Type": "application/json", "User-Agent": "TPM-MLX-Peer-Evict/0.5.0"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            logger.info(f"Cross-daemon eviction sent to port {port}: status={resp.status}")
    except Exception as e:
        logger.debug(f"Cross-daemon eviction notice (port {port} offline or idle: {e})")


async def get_or_load_image_engine(model_id: Optional[str] = None) -> Any:
    """
    Dynamically retrieves or initializes the MLX Image Engine.
    Cleans up old weights, performs mutual unload of text and 3D engines, and flushes Metal cache.
    """
    global image_engine, loaded_image_model_id, mlx_executor
    from tpm_mlx.image_engine import MLXImageEngine

    target_model = model_id or loaded_image_model_id or MLXImageEngine.DEFAULT_MODEL

    if image_engine is not None and loaded_image_model_id == target_model:
        return image_engine

    async with image_loading_lock:
        if image_engine is not None and loaded_image_model_id == target_model:
            return image_engine

        # Mutual Unload: Free active text and 3D engines before loading diffusion model
        if engine is not None:
            unload_text_engine()
        if three_d_engine is not None:
            unload_3d_engine()
        evict_peer_instance(2506)

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
            reclaim_metal_memory()

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

        # Mutual Unload: Free active image diffusion and 3D engines before loading text model
        if image_engine is not None:
            unload_image_engine()
        if three_d_engine is not None:
            unload_3d_engine()

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
            reclaim_metal_memory()

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

            reclaim_metal_memory()

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


def unload_embedding_engine() -> None:
    """
    Unloads active embedding engine and flushes Unified Memory and Metal GPU caches.
    """
    global embedding_engine, loaded_embedding_model_id
    if embedding_engine is not None:
        logger.info(f"Evicting embedding engine '{loaded_embedding_model_id}' from Unified Memory...")
        old_engine = embedding_engine
        embedding_engine = None
        loaded_embedding_model_id = None
        try:
            del old_engine.model
            del old_engine.processor
        except Exception:
            pass
        del old_engine
        reclaim_metal_memory()
        logger.info(f"Embedding engine evicted. Active Unified Memory: {get_active_memory_gb():.2f} GB")


class MLXEmbeddingEngine:
    """
    High-performance sovereign embedding engine powered by EmbeddingGemma 2 (4-bit Gemma 4 architecture).
    Provides MRL (Matryoshka Representation Learning) dimension slicing and unified memory acceleration.
    """
    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL):
        from mlx_vlm.embedding_loader import load_embedding_model
        from mlx_vlm.utils import load_processor, get_model_path
        logger.info(f"Loading embedding model: {model_id}...")
        self.model_id = model_id
        model_path = get_model_path(model_id)
        self.model = load_embedding_model(model_path)
        self.processor = load_processor(model_path)
        logger.info(f"Embedding model {model_id} loaded successfully.")

    def embed(self, texts: list[str], dimensions: Optional[int] = None) -> tuple[list[list[float]], int]:
        """
        Embeds a list of texts and returns (embeddings, total_tokens).
        Supports Matryoshka Representation Learning (MRL) dimension slicing and L2-normalization.
        """
        if not texts:
            return [], 0

        embed_dict = self.processor(text=texts, return_tensors="mlx", padding=True)
        res = self.model(**embed_dict)
        text_embeds = res.text_embeds

        # Apply Matryoshka Representation Learning (MRL) dimension truncation
        if dimensions is not None and dimensions < text_embeds.shape[-1]:
            text_embeds = text_embeds[:, :dimensions]
            # Re-normalize truncated embeddings to unit norm
            norms = mx.linalg.norm(text_embeds, axis=-1, keepdims=True)
            text_embeds = text_embeds / mx.maximum(norms, 1e-12)

        mx.eval(text_embeds)
        total_tokens = int(embed_dict["input_ids"].size)
        return text_embeds.tolist(), total_tokens


async def get_or_load_embedding_engine(model_id: Optional[str] = None) -> MLXEmbeddingEngine:
    """
    Retrieves or lazy-loads the requested sovereign embedding engine.
    """
    global embedding_engine, loaded_embedding_model_id
    target_id = model_id or DEFAULT_EMBEDDING_MODEL
    # Normalize aliases
    if target_id in ("google/embeddinggemma-2", "embeddinggemma-2", "text-embedding-3-small", "default"):
        target_id = DEFAULT_EMBEDDING_MODEL

    if embedding_engine is not None and loaded_embedding_model_id == target_id:
        return embedding_engine

    async with embedding_loading_lock:
        if embedding_engine is not None and loaded_embedding_model_id == target_id:
            return embedding_engine

        loop = asyncio.get_running_loop()

        def _load():
            return MLXEmbeddingEngine(target_id)

        eng = await loop.run_in_executor(mlx_executor, _load)
        embedding_engine = eng
        loaded_embedding_model_id = target_id
        return embedding_engine


async def get_or_load_3d_engine(model_id: Optional[str] = None) -> Any:
    """
    Dynamically retrieves or initializes the MLX 3D Engine.
    Performs mutual unload of text and 2D image diffusion engines to maximize unified memory headroom.
    """
    global three_d_engine, loaded_3d_model_id, mlx_executor
    from tpm_mlx.engine_3d import MLX3DEngine

    target_model = model_id or loaded_3d_model_id or DEFAULT_3D_MODEL

    if three_d_engine is not None and loaded_3d_model_id == target_model:
        return three_d_engine

    async with three_d_loading_lock:
        if three_d_engine is not None and loaded_3d_model_id == target_model:
            return three_d_engine

        # Mutual Unload: Free active text, image, and embedding engines
        if engine is not None:
            unload_text_engine()
        if image_engine is not None:
            unload_image_engine()
        if embedding_engine is not None:
            unload_embedding_engine()
        evict_peer_instance(2506)

        logger.info(f"Loading 3D engine with model: {target_model}...")

        old_3d = three_d_engine
        three_d_engine = None
        if old_3d is not None and loaded_3d_model_id != target_model:
            try:
                old_3d.unload()
            except Exception:
                pass
            del old_3d
            reclaim_metal_memory()

        def init_3d():
            eng = MLX3DEngine(model_id=target_model)
            eng.load_pipeline(target_model)
            return eng

        loop = asyncio.get_running_loop()
        new_3d_engine = await loop.run_in_executor(mlx_executor, init_3d)
        three_d_engine = new_3d_engine
        loaded_3d_model_id = target_model
        return three_d_engine


def get_image_engine() -> Any:
    """Synchronous accessor for active image engine (initializes default if absent)."""
    global image_engine, loaded_image_model_id
    from tpm_mlx.image_engine import MLXImageEngine

    if image_engine is None:
        if engine is not None:
            unload_text_engine()
        if three_d_engine is not None:
            unload_3d_engine()
        image_engine = MLXImageEngine(lazy=False)
        loaded_image_model_id = MLXImageEngine.DEFAULT_MODEL
    return image_engine


def get_3d_engine() -> Any:
    """Synchronous accessor for active 3D engine (initializes default if absent)."""
    global three_d_engine, loaded_3d_model_id
    from tpm_mlx.engine_3d import MLX3DEngine

    if three_d_engine is None:
        if engine is not None:
            unload_text_engine()
        if image_engine is not None:
            unload_image_engine()
        eng = MLX3DEngine(model_id=DEFAULT_3D_MODEL)
        eng.load_pipeline(DEFAULT_3D_MODEL)
        three_d_engine = eng
        loaded_3d_model_id = DEFAULT_3D_MODEL
    return three_d_engine


# Backwards compatibility aliases
_get_or_load_image_engine = get_or_load_image_engine
_load_engine = load_engine
_get_or_load_embedding_engine = get_or_load_embedding_engine
_get_or_load_3d_engine = get_or_load_3d_engine

