# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
OpenAI-compatible schema definitions for /v1/embeddings endpoint.
Supports Matryoshka Representation Learning (MRL) dimension truncation,
single-string and batch inputs, and multimodal embedding representations.
"""

from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: Optional[str] = "google/embeddinggemma-2"
    input: Union[str, List[str], List[Dict[str, Any]]]
    dimensions: Optional[int] = Field(default=None, description="Optional MRL dimension truncation (e.g. 768, 512, 256, 128)")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format: 'float' or 'base64'")
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
