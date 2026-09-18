# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import os
import time
import json
import uuid
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tpm_mlx.engine import MLXEngine
from tpm_mlx.utils import get_logger, get_cached_models, install_safe_streams

install_safe_streams()
logger = get_logger("server")

## MLX requires GPU stream affinity. We use a single dedicated thread executor for all MLX operations.
mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_thread")

# Global engine instance and currently loaded model name
engine: Optional[MLXEngine] = None
loaded_model_id: Optional[str] = None
loaded_draft_model_id: Optional[str] = None
model_loading_lock = asyncio.Lock()

# Global image engine instance (lazy loaded on demand)
image_engine: Optional[Any] = None
loaded_image_model_id: Optional[str] = None
image_loading_lock = asyncio.Lock()

# Directory for persisted generated images
images_dir = Path(__file__).parent / "static" / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Global default max KV size
default_max_kv_size = 4096

app = FastAPI(
    title="TPM-MLX Server",
    description="Optimized Apple Silicon Inference Engine API Server",
    version="0.2.0"
)

# Mount static images directory for generated assets
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

# Enable CORS for easy cross-origin integrations (e.g. Continue, Page playgrounds)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input Schemas
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


def _extract_text_and_images(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Extracts text messages and PIL images from OpenAI multimodal messages format."""
    formatted_messages = []
    images = []
    
    for m in messages:
        if isinstance(m.content, str):
            formatted_messages.append({"role": m.role, "content": m.content})
        elif isinstance(m.content, list):
            text_parts = []
            for part in m.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_info = part.get("image_url", {})
                        img_url = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                        if img_url:
                            try:
                                from PIL import Image
                                import io, base64
                                if img_url.startswith("data:image"):
                                    header, base64_data = img_url.split(",", 1)
                                    img_bytes = base64.b64decode(base64_data)
                                    img = Image.open(io.BytesIO(img_bytes))
                                    images.append(img)
                                elif img_url.startswith("http"):
                                    import requests
                                    resp = requests.get(img_url, timeout=10)
                                    img = Image.open(io.BytesIO(resp.content))
                                    images.append(img)
                            except Exception as ex:
                                logger.warning(f"Failed to load image from {img_url[:30]}...: {ex}")
            formatted_messages.append({"role": m.role, "content": " ".join(text_parts)})
        else:
            formatted_messages.append({"role": m.role, "content": str(m.content)})
            
    return formatted_messages, images


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    reasoning: Optional[bool] = Field(default=None, description="Toggles outputting reasoning <think> blocks")


class LoadModelRequest(BaseModel):
    model: str
    draft_model: Optional[str] = None
    max_kv_size: Optional[int] = None
    enable_mtp: Optional[bool] = True
    num_draft_tokens: Optional[int] = None


class ImageGenerationApiRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    model: Optional[str] = None
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    steps: int = Field(default=4, ge=1, le=50)
    guidance: Optional[float] = None
    seed: Optional[int] = None
    response_format: str = "url"  # "url" or "b64_json"
    auto_expand: Optional[bool] = None
    context: Optional[str] = None


class ImageEditApiRequest(BaseModel):
    image: str  # URL, relative path, or base64 data URI
    prompt: str
    model: Optional[str] = None
    size: Optional[str] = None
    steps: int = Field(default=4, ge=1, le=50)
    guidance: Optional[float] = 2.5
    seed: Optional[int] = None
    response_format: str = "url"  # "url" or "b64_json"
    auto_expand: Optional[bool] = None
    context: Optional[str] = None


class LoadImageModelRequest(BaseModel):
    model: str


# Helper to dynamically get or load the image engine on demand
async def _get_or_load_image_engine(model_id: Optional[str] = None):
    global image_engine, loaded_image_model_id, mlx_executor
    from tpm_mlx.image_engine import MLXImageEngine

    target_model = model_id or loaded_image_model_id or MLXImageEngine.DEFAULT_MODEL

    if image_engine is not None and loaded_image_model_id == target_model:
        return image_engine

    async with image_loading_lock:
        if image_engine is not None and loaded_image_model_id == target_model:
            return image_engine

        logger.info(f"Loading image engine with model: {target_model}...")

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


# Helper to dynamically load a model in the server
async def _load_engine(
    model_id: str, 
    max_kv_size: int,
    draft_model: Optional[str] = None,
    enable_mtp: bool = True,
    num_draft_tokens: Optional[int] = None,
):
    global engine, loaded_model_id, loaded_draft_model_id, mlx_executor
    async with model_loading_lock:
        logger.info(f"Loading model: {model_id} (KV Cache Size: {max_kv_size}, Draft: {draft_model}, MTP: {enable_mtp})...")
        start_time = time.perf_counter()
        
        # Reset and refresh dedicated worker to cancel stale queue backlogs immediately
        old_exec = mlx_executor
        mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_thread")
        try:
            old_exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
            
        # Clean up existing engine and release Metal GPU buffers before loading new model
        old_engine = engine
        engine = None
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
            import gc, mlx.core as mx
            gc.collect()
            mx.clear_cache()
            if hasattr(mx, "metal"):
                mx.metal.clear_cache()
                
            try:
                eng = MLXEngine(
                    model_path_or_id=model_id, 
                    max_kv_size=max_kv_size,
                    draft_model_path_or_id=draft_model if draft_model and draft_model.strip() else None,
                    enable_mtp=enable_mtp,
                    num_draft_tokens=num_draft_tokens,
                )
            except BrokenPipeError:
                logger.warning("Caught BrokenPipeError during MLX engine load; retrying once...")
                time.sleep(0.5)
                eng = MLXEngine(
                    model_path_or_id=model_id, 
                    max_kv_size=max_kv_size,
                    draft_model_path_or_id=draft_model if draft_model and draft_model.strip() else None,
                    enable_mtp=enable_mtp,
                    num_draft_tokens=num_draft_tokens,
                )
            
            gc.collect()
            mx.clear_cache()
            if hasattr(mx, "metal"):
                mx.metal.clear_cache()
                
            return eng

        loop = asyncio.get_running_loop()
        new_engine = await loop.run_in_executor(mlx_executor, init_engine)
        
        engine = new_engine
        loaded_model_id = model_id
        loaded_draft_model_id = draft_model if draft_model and draft_model.strip() else None
        duration = time.perf_counter() - start_time
        logger.info(f"Successfully loaded {model_id} [Speculation: {engine.speculation_mode.upper()}] in {duration:.2f}s")


@app.on_event("startup")
async def startup_event():
    # Attempt to load default model if specified in environment
    default_model = os.environ.get("TPM_DEFAULT_MODEL")
    draft_model = os.environ.get("TPM_DEFAULT_DRAFT_MODEL")
    kv_size = int(os.environ.get("TPM_MAX_KV_SIZE", str(default_max_kv_size)))
    enable_mtp = os.environ.get("TPM_ENABLE_MTP", "True").lower() == "true"
    num_draft = int(os.environ.get("TPM_NUM_DRAFT_TOKENS")) if os.environ.get("TPM_NUM_DRAFT_TOKENS") else None
    
    if default_model and default_model.strip().lower() not in ("none", "null", "false", ""):
        try:
            await _load_engine(
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
            await _get_or_load_image_engine(default_image_model)
        except Exception as e:
            logger.error(f"Failed to load default image model {default_image_model} on startup: {e}")


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_playground():
    """Serves the static Web Playground HTML."""
    static_dir = Path(__file__).parent / "static"
    playground_path = static_dir / "playground.html"
    
    if not playground_path.exists():
        return HTMLResponse(
            content="<h3>Playground HTML not found. Run building steps.</h3>", 
            status_code=404
        )
        
    with open(playground_path, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)


def is_image_repo(repo_id: Optional[str]) -> bool:
    if not repo_id:
        return False
    lower = repo_id.lower()
    return any(k in lower for k in ("flux", "diffusion", "bonsai", "klein", "sdxl", "stable-diffusion", "z_image", "z-image", "zimage"))


@app.get("/v1/models")
async def list_models():
    """
    Returns list of loaded models and Hugging Face cached models.
    """
    data = []
    seen_ids = set()

    # 1. Add currently loaded text/vision LLM or image model if available
    is_curr_img = is_image_repo(loaded_model_id) if loaded_model_id else False
    if loaded_model_id:
        seen_ids.add(loaded_model_id)
        data.append({
            "id": loaded_model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tpm-mlx",
            "active": True,
            "active_type": "image" if is_curr_img else "llm",
            "is_draft": False,
            "is_image": is_curr_img,
            "backend": "image" if is_curr_img else getattr(engine, "backend", "llm"),
            "max_kv_size": getattr(engine, "max_kv_size", default_max_kv_size),
            "speculation_mode": "none" if is_curr_img else getattr(engine, "speculation_mode", "none"),
            "has_mtp": False if is_curr_img else getattr(engine, "has_mtp", False),
            "num_draft_tokens": 0 if is_curr_img else getattr(engine, "num_draft_tokens", 0),
            "draft_model": None if is_curr_img else loaded_draft_model_id,
        })

    # 2. Retrieve verified cached models on disk
    cached = get_cached_models()
    for item in cached:
        repo_id = item["repo_id"]
        if repo_id in seen_ids:
            continue
        seen_ids.add(repo_id)

        is_active_image = bool(loaded_image_model_id and repo_id == loaded_image_model_id)
        is_active_draft = bool(loaded_draft_model_id and repo_id == loaded_draft_model_id)

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
        "active_model": loaded_model_id,
        "active_draft_model": None if is_curr_img else loaded_draft_model_id,
        "active_image_model": loaded_image_model_id,
        "speculation_mode": "none" if is_curr_img else (getattr(engine, "speculation_mode", "none") if engine else "none"),
        "has_mtp": False if is_curr_img else (getattr(engine, "has_mtp", False) if engine else False),
        "num_draft_tokens": 0 if is_curr_img else (getattr(engine, "num_draft_tokens", 0) if engine else 0),
        "backend": "image" if is_curr_img else (getattr(engine, "backend", "llm") if engine else "llm"),
    }


@app.post("/v1/load_model")
async def load_model_endpoint(req: LoadModelRequest):
    """
    Endpoint to load/switch models dynamically from the playground or API.
    Auto-detects image vs text models and routes accordingly.
    """
    global default_max_kv_size, loaded_model_id, engine, loaded_draft_model_id
    
    if req.model.strip().lower() in ("none", "null", "unload"):
        old_engine = engine
        engine = None
        loaded_model_id = None
        loaded_draft_model_id = None
        if old_engine is not None:
            try:
                del old_engine.model
                del old_engine.processor
                del old_engine.drafter
            except Exception:
                pass
            del old_engine
        import gc
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

    if is_image_repo(req.model):
        await _get_or_load_image_engine(req.model)
        loaded_model_id = req.model
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

    kv_size = req.max_kv_size or default_max_kv_size
    enable_mtp = req.enable_mtp if req.enable_mtp is not None else True
    try:
        await _load_engine(
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
            "draft_model": loaded_draft_model_id,
            "backend": engine.backend if engine else "llm",
            "speculation_mode": engine.speculation_mode if engine else "none",
            "has_mtp": engine.has_mtp if engine else False,
            "num_draft_tokens": engine.num_draft_tokens if engine else 0,
            "is_image": False,
        }
    except Exception as e:
        logger.error(f"Error loading model {req.model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with reasoning filtering and performance stats.
    """
    global engine, loaded_model_id
    current_engine = engine
    current_model_id = loaded_model_id
    
    if current_engine is None:
        raise HTTPException(
            status_code=400, 
            detail="No model is loaded. Please load a model using /v1/load_model first."
        )
        
    # Resolve reasoning flag
    if req.reasoning is not None:
        show_reasoning = req.reasoning
    else:
        show_reasoning = os.environ.get("TPM_DEFAULT_REASONING", "False").lower() == "true"
        
    # Standard OpenAI Chat template format mapping with multimodal image extraction
    formatted_messages, images = _extract_text_and_images(req.messages)
    
    try:
        if hasattr(current_engine, "processor") and hasattr(current_engine.processor, "apply_chat_template"):
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(current_engine.processor, current_engine.model.config, formatted_messages)
        else:
            prompt = current_engine.tokenizer.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
    except Exception as e:
        from tpm_mlx.utils import apply_chat_template_fallback
        prompt = apply_chat_template_fallback(formatted_messages, current_engine.tokenizer)
        logger.warning(f"Could not apply tokenizer template ({e}), using fallback formatting.")
        
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Generate in Executor to avoid blocking FastAPI server main loop
    loop = asyncio.get_running_loop()
    
    if req.stream:
        async def event_generator():
            queue = asyncio.Queue(maxsize=16)
            
            def producer():
                try:
                    for response in current_engine.generate_stream(
                        prompt=prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        show_reasoning=show_reasoning,
                        images=images if images else None,
                    ):
                        asyncio.run_coroutine_threadsafe(queue.put(response), loop).result()
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
                except Exception as ex:
                    logger.error(f"Error in stream producer thread: {ex}")
                    asyncio.run_coroutine_threadsafe(queue.put(ex), loop).result()
            
            # Start generator in executor thread
            gen_task = loop.run_in_executor(mlx_executor, producer)
            
            # Read from async queue
            prompt_tokens_count = 0
            completion_tokens_count = 0
            generation_tps = 0.0
            prompt_tps = 0.0
            peak_mem = 0.0
            ttft = 0.0
            start_time = time.perf_counter()
            
            metrics_sent = False
            
            while True:
                item = await queue.get()
                if item is None:
                    # Finalize stream and emit metrics chunk if not already sent
                    if not metrics_sent and completion_tokens_count > 0:
                        final_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": current_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens_count,
                                "completion_tokens": completion_tokens_count,
                                "total_tokens": prompt_tokens_count + completion_tokens_count
                            },
                            "tpm_metrics": {
                                "tps": round(generation_tps, 2),
                                "ttft_ms": round(ttft, 2),
                                "prompt_tps": round(prompt_tps, 2),
                                "peak_memory_gb": round(peak_mem, 2),
                                "prompt_tokens": prompt_tokens_count,
                                "generation_tokens": completion_tokens_count,
                                "speculation_mode": current_engine.speculation_mode if current_engine else "none",
                                "acceptance_rate": round(current_engine.speculation_stats.acceptance_rate, 4) if current_engine else 0.0,
                                "draft_tokens_total": current_engine.speculation_stats.draft_tokens_total if current_engine else 0,
                                "accepted_tokens_total": current_engine.speculation_stats.accepted_tokens_total if current_engine else 0,
                            }
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                    break
                    
                if isinstance(item, Exception):
                    yield f"data: {{\"error\": \"{str(item)}\"}}\n\n"
                    break
                    
                completion_tokens_count = item.generation_tokens
                prompt_tokens_count = item.prompt_tokens
                generation_tps = item.generation_tps
                prompt_tps = item.prompt_tps
                peak_mem = item.peak_memory
                
                if ttft == 0.0 and (completion_tokens_count > 0 or bool(item.text)):
                    ttft = max((time.perf_counter() - start_time) * 1000.0, 1.0)
                
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": current_model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": item.text},
                            "finish_reason": item.finish_reason
                        }
                    ]
                }
                
                # If it is the final token, append metrics and usage metadata
                if item.finish_reason is not None:
                    metrics_sent = True
                    chunk["usage"] = {
                        "prompt_tokens": prompt_tokens_count,
                        "completion_tokens": completion_tokens_count,
                        "total_tokens": prompt_tokens_count + completion_tokens_count
                    }
                    chunk["tpm_metrics"] = {
                        "tps": round(generation_tps, 2),
                        "ttft_ms": round(ttft, 2),
                        "prompt_tps": round(prompt_tps, 2),
                        "peak_memory_gb": round(peak_mem, 2),
                        "prompt_tokens": prompt_tokens_count,
                        "generation_tokens": completion_tokens_count,
                        "speculation_mode": current_engine.speculation_mode if current_engine else "none",
                        "acceptance_rate": round(current_engine.speculation_stats.acceptance_rate, 4) if current_engine else 0.0,
                        "draft_tokens_total": current_engine.speculation_stats.draft_tokens_total if current_engine else 0,
                        "accepted_tokens_total": current_engine.speculation_stats.accepted_tokens_total if current_engine else 0,
                    }
                    
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    else:
        # Non-streaming implementation: consume full stream in thread executor
        def consume_generator():
            responses = []
            for response in current_engine.generate_stream(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                show_reasoning=show_reasoning,
                images=images if images else None,
            ):
                responses.append(response)
            return responses
            
        try:
            responses = await loop.run_in_executor(mlx_executor, consume_generator)
        except asyncio.CancelledError:
            raise HTTPException(status_code=503, detail="Generation interrupted due to model reload.")
            
        if not responses:
            raise HTTPException(status_code=500, detail="Model generated zero responses")
            
        # Compile full response text and stats
        full_text = "".join(r.text for r in responses)
        last_resp = responses[-1]
        
        # Calculate TTFT for non-streaming
        ttft_ms = (last_resp.prompt_tokens / max(last_resp.prompt_tps, 1e-6) * 1000.0) if last_resp.prompt_tps > 0 else 1.0
        
        response_json = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": current_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text
                    },
                    "finish_reason": last_resp.finish_reason or "stop"
                }
            ],
            "usage": {
                "prompt_tokens": last_resp.prompt_tokens,
                "completion_tokens": last_resp.generation_tokens,
                "total_tokens": last_resp.prompt_tokens + last_resp.generation_tokens
            },
            "tpm_metrics": {
                "tps": round(last_resp.generation_tps, 2),
                "ttft_ms": round(ttft_ms, 2),
                "prompt_tps": round(last_resp.prompt_tps, 2),
                "peak_memory_gb": round(last_resp.peak_memory, 2),
                "speculation_mode": current_engine.speculation_mode,
                "acceptance_rate": round(current_engine.speculation_stats.acceptance_rate, 4),
                "draft_tokens_total": current_engine.speculation_stats.draft_tokens_total,
                "accepted_tokens_total": current_engine.speculation_stats.accepted_tokens_total,
            }
        }
        
        return JSONResponse(content=response_json)


@app.post("/v1/load_image_model")
async def load_image_model_endpoint(req: LoadImageModelRequest):
    """Dynamically loads or switches the active image generation model."""
    await _get_or_load_image_engine(req.model)
    return JSONResponse(content={"status": "ok", "loaded_image_model": req.model})


@app.post("/v1/images/generations")
async def generate_images_endpoint(req: ImageGenerationApiRequest):
    """
    OpenAI-compatible image generation endpoint.
    Supports resolution, steps, seed, response_format (url / b64_json),
    and intelligent context distillation via the active text LLM engine.
    """
    img_eng = await _get_or_load_image_engine(req.model)
    
    effective_prompt = req.prompt
    revised_prompt: Optional[str] = None
    
    # Context Distillation: Strictly bypass if auto_expand is False. Only distill if True or (None with context/length)
    should_distill = False
    if req.auto_expand is True:
        should_distill = True
    elif req.auto_expand is None and (bool(req.context) or len(req.prompt) > 300):
        should_distill = True

    if should_distill and engine is not None:
        try:
            loop = asyncio.get_running_loop()
            def distill():
                distill_prompt_text = (
                    f"<|im_start|>system\nYou are an expert visual prompt director. "
                    f"Analyze the context and request, then synthesize a single dense, photorealistic "
                    f"image generation prompt focusing on subjects, lighting, colors, mood, and composition (max 80 words). "
                    f"Output ONLY the prompt text.<|im_end|>\n"
                    f"<|im_start|>user\nContext:\n{req.context or ''}\n\nRequest: {req.prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                responses = list(engine.generate_stream(distill_prompt_text, max_tokens=200, temperature=0.3, show_reasoning=True))
                text = "".join(r.text for r in responses).strip()
                if "</think>" in text:
                    text = text.split("</think>", 1)[-1].strip()
                elif "<channel|>" in text:
                    text = text.split("<channel|>", 1)[-1].strip()
                if ":" in text and len(text.split(":", 1)[0]) < 60:
                    prefix = text.split(":", 1)[0].lower()
                    if any(k in prefix for k in ("prompt", "output", "description", "image")):
                        text = text.split(":", 1)[1].strip()
                mx.synchronize()
                return text
                
            revised = await loop.run_in_executor(mlx_executor, distill)
            if revised and len(revised) > 10:
                effective_prompt = revised
                revised_prompt = revised
                logger.info(f"Distilled visual prompt: '{effective_prompt[:70]}...'")
        except Exception as ex:
            logger.warning(f"Context distillation skipped: {ex}")

    from tpm_mlx.image_engine import MLXImageEngine
    detected_size = MLXImageEngine.extract_resolution_from_text(req.prompt)
    effective_size = detected_size or req.size or "1024x1024"

    # Generate image on the dedicated mlx_executor thread
    loop = asyncio.get_running_loop()
    def run_gen():
        filename = f"gen_{int(time.time())}_{random.randint(1000, 9999)}.png"
        filepath = images_dir / filename
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
        result, fname = await loop.run_in_executor(mlx_executor, run_gen)
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


@app.post("/v1/upload_image")
async def upload_image_endpoint(file: UploadFile = File(...)):
    """Uploads a local user image for Image-to-Image editing."""
    uploads_dir = images_dir / "uploads"
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


@app.post("/v1/images/edits")
async def edit_images_endpoint(req: ImageEditApiRequest):
    """
    OpenAI-compatible image editing endpoint.
    Modifies an input reference image according to a natural language prompt.
    """
    img_eng = await _get_or_load_image_engine(req.model)
    
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

    if should_distill and engine is not None:
        try:
            loop = asyncio.get_running_loop()
            def distill():
                distill_prompt_text = (
                    f"<|im_start|>system\nYou are an expert visual prompt director for image editing. "
                    f"Analyze the edit request and context, then synthesize a concise, precise image modification prompt "
                    f"focusing on the exact changes to be made (clothing, colors, lighting, objects) while retaining unchanged elements (max 60 words). "
                    f"Output ONLY the prompt text.<|im_end|>\n"
                    f"<|im_start|>user\nContext:\n{req.context or ''}\n\nEdit Request: {req.prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                responses = list(engine.generate_stream(distill_prompt_text, max_tokens=150, temperature=0.3, show_reasoning=True))
                text = "".join(r.text for r in responses).strip()
                if "</think>" in text:
                    text = text.split("</think>", 1)[-1].strip()
                elif "<channel|>" in text:
                    text = text.split("<channel|>", 1)[-1].strip()
                if ":" in text and len(text.split(":", 1)[0]) < 60:
                    prefix = text.split(":", 1)[0].lower()
                    if any(k in prefix for k in ("prompt", "output", "description", "image")):
                        text = text.split(":", 1)[1].strip()
                mx.synchronize()
                return text
                
            revised = await loop.run_in_executor(mlx_executor, distill)
            if revised and len(revised) > 10:
                effective_prompt = revised
                revised_prompt = revised
                logger.info(f"Distilled visual edit prompt: '{effective_prompt[:70]}...'")
        except Exception as ex:
            logger.warning(f"Context distillation skipped: {ex}")

    from tpm_mlx.image_engine import MLXImageEngine
    detected_size = MLXImageEngine.extract_resolution_from_text(req.prompt)
    effective_size = detected_size or req.size

    loop = asyncio.get_running_loop()
    def run_edit():
        filename = f"edit_{int(time.time())}_{random.randint(1000, 9999)}.png"
        filepath = images_dir / filename
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
        result, fname = await loop.run_in_executor(mlx_executor, run_edit)
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
