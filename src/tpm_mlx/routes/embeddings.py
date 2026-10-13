# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
OpenAI-compatible /v1/embeddings endpoint powered by EmbeddingGemma 2.
Supports Matryoshka Representation Learning (MRL) dimension slicing (768, 512, 256, 128)
and sovereign local embedding inference on Apple Silicon.
"""

import logging
import asyncio
from fastapi import APIRouter, HTTPException

from tpm_mlx import state
from tpm_mlx.schemas.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)

logger = logging.getLogger("tpm-mlx.routes.embeddings")
router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generates OpenAI-compatible vector embeddings for strings or lists of strings.
    Supports MRL dimension truncation via `request.dimensions`.
    """
    raw_input = request.input
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        if not raw_input:
            raise HTTPException(status_code=400, detail="Input list cannot be empty.")
        if isinstance(raw_input[0], str):
            texts = raw_input
        elif isinstance(raw_input[0], dict):
            texts = [item.get("text", "") for item in raw_input]
        else:
            raise HTTPException(status_code=400, detail="Unsupported input item type.")
    else:
        raise HTTPException(status_code=400, detail="Input must be a string or list of strings.")

    # Validate MRL dimension limits if provided
    if request.dimensions is not None:
        if request.dimensions <= 0 or request.dimensions > 768:
            raise HTTPException(
                status_code=400,
                detail=f"Requested dimension {request.dimensions} invalid. Must be between 1 and 768.",
            )

    try:
        engine = await state.get_or_load_embedding_engine(request.model)
        loop = asyncio.get_running_loop()
        embeddings, total_tokens = await loop.run_in_executor(
            state.mlx_executor,
            engine.embed,
            texts,
            request.dimensions,
        )

        data = [
            EmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            object="list",
            data=data,
            model=engine.model_id,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
