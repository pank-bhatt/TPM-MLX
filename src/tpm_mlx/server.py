# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import os
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from tpm_mlx.engine import MLXEngine
from tpm_mlx.utils import get_logger, get_cached_models

logger = get_logger("server")

## MLX requires GPU stream affinity. We use a single dedicated thread executor for all MLX operations.
mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx_thread")

# Global engine instance and currently loaded model name
engine: Optional[MLXEngine] = None
loaded_model_id: Optional[str] = None
loaded_draft_model_id: Optional[str] = None
model_loading_lock = asyncio.Lock()

# Global default max KV size
default_max_kv_size = 4096

app = FastAPI(
    title="TPM-MLX Server",
    description="Optimized Apple Silicon Inference Engine API Server",
    version="0.2.0"
)

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
    
    if default_model:
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


@app.get("/v1/models")
async def list_models():
    """
    Returns list of loaded models and Hugging Face cached models.
    """
    data = []
    
    def is_draft_repo(repo_id: str) -> bool:
        lower = repo_id.lower()
        return "assistant" in lower or "-mtp" in lower or "_mtp" in lower or "drafter" in lower
    
    # 1. Add currently loaded model if available
    if loaded_model_id:
        data.append({
            "id": loaded_model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tpm-mlx",
            "active": True,
            "is_draft": False,
            "backend": getattr(engine, "backend", "llm"),
            "max_kv_size": getattr(engine, "max_kv_size", default_max_kv_size),
            "speculation_mode": getattr(engine, "speculation_mode", "none"),
            "has_mtp": getattr(engine, "has_mtp", False),
            "num_draft_tokens": getattr(engine, "num_draft_tokens", 0),
            "draft_model": loaded_draft_model_id,
        })
        
    # 2. Retrieve cached models on disk
    cached = get_cached_models()
    for item in cached:
        if item["repo_id"] != loaded_model_id:
            data.append({
                "id": item["repo_id"],
                "object": "model",
                "created": int(item["last_modified"]),
                "owned_by": "huggingface",
                "active": False,
                "is_draft": is_draft_repo(item["repo_id"]),
                "size_bytes": item["size_on_disk"]
            })
            
    return {
        "object": "list", 
        "data": data,
        "active_model": loaded_model_id,
        "active_draft_model": loaded_draft_model_id,
        "speculation_mode": getattr(engine, "speculation_mode", "none") if engine else "none",
        "has_mtp": getattr(engine, "has_mtp", False) if engine else False,
        "num_draft_tokens": getattr(engine, "num_draft_tokens", 0) if engine else 0,
        "backend": getattr(engine, "backend", "llm") if engine else "llm",
    }


@app.post("/v1/load_model")
async def load_model_endpoint(req: LoadModelRequest):
    """
    Endpoint to load/switch models dynamically from the playground or API.
    """
    global default_max_kv_size
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
                
                if completion_tokens_count == 1:
                    ttft = (time.perf_counter() - start_time) * 1000.0  # ms
                
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
        
        # Calculate TTFT
        ttft_ms = 0.0
        
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
