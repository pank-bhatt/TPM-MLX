# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
OpenAI-compatible /v1/chat/completions endpoint.
Supports streaming SSE chunks with real-time performance telemetry,
speculative token acceptance statistics, reasoning <think> flags, and multimodal vision.
"""

import os
import json
import time
import uuid
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from tpm_mlx import state
from tpm_mlx.schemas.chat import ChatCompletionRequest
from tpm_mlx.vlm import extract_text_and_images
from tpm_mlx.utils import apply_chat_template_fallback

logger = logging.getLogger("tpm-mlx.routes.chat")
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with reasoning filtering and performance stats.
    """
    current_engine = state.engine
    current_model_id = state.loaded_model_id

    if current_engine is None:
        raise HTTPException(
            status_code=400,
            detail="No model is loaded. Please load a model using /v1/load_model first.",
        )

    # Resolve reasoning flag
    if req.reasoning is not None:
        show_reasoning = req.reasoning
    else:
        show_reasoning = os.environ.get("TPM_DEFAULT_REASONING", "False").lower() == "true"

    # Standard OpenAI Chat template format mapping with multimodal image extraction
    formatted_messages, images = extract_text_and_images(req.messages)

    try:
        if hasattr(current_engine, "processor") and hasattr(current_engine.processor, "apply_chat_template"):
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(current_engine.processor, current_engine.model.config, formatted_messages)
        else:
            prompt = current_engine.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as e:
        prompt = apply_chat_template_fallback(formatted_messages, current_engine.tokenizer)
        logger.warning(f"Could not apply tokenizer template ({e}), using fallback formatting.")

    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

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
            loop.run_in_executor(state.mlx_executor, producer)

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
            responses = await loop.run_in_executor(state.mlx_executor, consume_generator)
        except asyncio.CancelledError:
            raise HTTPException(status_code=503, detail="Generation interrupted due to model reload.")

        if not responses:
            raise HTTPException(status_code=500, detail="Model generated zero responses")

        full_text = "".join(r.text for r in responses)
        last_resp = responses[-1]
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
