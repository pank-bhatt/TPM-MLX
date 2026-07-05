# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import os
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from tpm_mlx.engine import MLXEngine
from tpm_mlx.utils import get_logger, get_cached_models

logger = get_logger("server")

# Global engine instance and currently loaded model name
engine: Optional[MLXEngine] = None
loaded_model_id: Optional[str] = None
model_loading_lock = asyncio.Lock()

# Global default max KV size
default_max_kv_size = 4096

app = FastAPI(
    title="TPM-MLX Server",
    description="Optimized Apple Silicon Inference Engine API Server",
    version="0.1.0"
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
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    reasoning: bool = Field(default=True, description="Toggles outputting reasoning <think> blocks")


class LoadModelRequest(BaseModel):
    model: str
    max_kv_size: Optional[int] = None


# Helper to dynamically load a model in the server
async def _load_engine(model_id: str, max_kv_size: int):
    global engine, loaded_model_id
    async with model_loading_lock:
        logger.info(f"Loading model: {model_id} (KV Cache Size: {max_kv_size})...")
        start_time = time.perf_counter()
        
        # Instantiate engine (CPU-GPU operations are synchronous in MLX but we run in executor if needed)
        # To avoid blocking the async loop, we run synchronous init in a thread executor
        def init_engine():
            return MLXEngine(model_path_or_id=model_id, max_kv_size=max_kv_size)
            
        loop = asyncio.get_running_loop()
        new_engine = await loop.run_in_executor(None, init_engine)
        
        engine = new_engine
        loaded_model_id = model_id
        duration = time.perf_counter() - start_time
        logger.info(f"Successfully loaded {model_id} in {duration:.2f}s")


@app.on_event("startup")
async def startup_event():
    # Attempt to load default model if specified in environment
    default_model = os.environ.get("TPM_DEFAULT_MODEL")
    kv_size = int(os.environ.get("TPM_MAX_KV_SIZE", str(default_max_kv_size)))
    if default_model:
        try:
            await _load_engine(default_model, kv_size)
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
    
    # 1. Add currently loaded model if available
    if loaded_model_id:
        data.append({
            "id": loaded_model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tpm-mlx",
            "active": True,
            "max_kv_size": getattr(engine, "max_kv_size", default_max_kv_size)
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
                "size_bytes": item["size_on_disk"]
            })
            
    return {"object": "list", "data": data}


@app.post("/v1/load_model")
async def load_model_endpoint(req: LoadModelRequest):
    """
    Endpoint to load/switch models dynamically from the playground or API.
    """
    global default_max_kv_size
    kv_size = req.max_kv_size or default_max_kv_size
    try:
        await _load_engine(req.model, kv_size)
        return {
            "status": "success",
            "message": f"Successfully loaded model {req.model}",
            "model": req.model
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
    
    if engine is None:
        raise HTTPException(
            status_code=400, 
            detail="No model is loaded. Please load a model using /v1/load_model first."
        )
        
    # Standard OpenAI Chat template format mapping
    formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    try:
        # Use tokenizer to apply chat template
        prompt = engine.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback manual prompt construction if tokenizer template fails
        from tpm_mlx.utils import apply_chat_template_fallback
        prompt = apply_chat_template_fallback(formatted_messages, engine.tokenizer)
        logger.warning(f"Could not apply tokenizer template ({e}), using fallback formatting.")
        
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Generate in Executor to avoid blocking FastAPI server main loop
    loop = asyncio.get_running_loop()
    
    if req.stream:
        async def event_generator():
            # Run stream generation loop
            # Define synchronous generation call wrapper
            def run_gen():
                return list(
                    engine.generate_stream(
                        prompt=prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        show_reasoning=req.reasoning
                    )
                )
                
            # Unfortunately calling list(generator) defeats the streaming purpose,
            # so we must consume it iteratively in the thread executor.
            # We use an queue to bridge thread generator and async generator.
            queue = asyncio.Queue(maxsize=16)
            
            def producer():
                try:
                    for response in engine.generate_stream(
                        prompt=prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        show_reasoning=req.reasoning
                    ):
                        # Use run_coroutine_threadsafe to push to async queue
                        asyncio.run_coroutine_threadsafe(queue.put(response), loop).result()
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
                except Exception as ex:
                    logger.error(f"Error in stream producer thread: {ex}")
                    asyncio.run_coroutine_threadsafe(queue.put(ex), loop).result()
            
            # Start generator in executor thread
            gen_task = loop.run_in_executor(None, producer)
            
            # Read from async queue
            prompt_tokens_count = 0
            completion_tokens_count = 0
            generation_tps = 0.0
            prompt_tps = 0.0
            peak_mem = 0.0
            ttft = 0.0
            start_time = time.perf_counter()
            
            while True:
                item = await queue.get()
                if item is None:
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
                    "model": loaded_model_id,
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
                    chunk["usage"] = {
                        "prompt_tokens": prompt_tokens_count,
                        "completion_tokens": completion_tokens_count,
                        "total_tokens": prompt_tokens_count + completion_tokens_count
                    }
                    chunk["tpm_metrics"] = {
                        "tps": round(generation_tps, 2),
                        "ttft_ms": round(ttft, 2),
                        "prompt_tps": round(prompt_tps, 2),
                        "peak_memory_gb": round(peak_mem, 2)
                    }
                    
                yield f"data: {json.dumps(chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    else:
        # Non-streaming implementation: consume full stream in thread executor
        def consume_generator():
            responses = []
            for response in engine.generate_stream(
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                show_reasoning=req.reasoning
            ):
                responses.append(response)
            return responses
            
        responses = await loop.run_in_executor(None, consume_generator)
        if not responses:
            raise HTTPException(status_code=500, detail="Model generated zero responses")
            
        # Compile full response text and stats
        full_text = "".join(r.text for r in responses)
        last_resp = responses[-1]
        
        # Calculate TTFT (rough estimate for non-streaming)
        ttft_ms = 0.0 # Not highly relevant for sync non-stream, but we provide it
        
        response_json = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": loaded_model_id,
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
                "peak_memory_gb": round(last_resp.peak_memory, 2)
            }
        }
        
        return response_json
