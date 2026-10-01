# Copyright © 2026 TPM-MLX Authors. All rights reserved.

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LoadModelRequest(BaseModel):
    model: str
    draft_model: Optional[str] = None
    max_kv_size: Optional[int] = None
    enable_mtp: Optional[bool] = True
    num_draft_tokens: Optional[int] = None
    async_load: Optional[bool] = False


class LoadImageModelRequest(BaseModel):
    model: str


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1714000000
    owned_by: str = "tpm-mlx"
    active: bool = False
    active_type: Optional[str] = None
    is_draft: bool = False
    is_image: bool = False
    max_kv_size: Optional[int] = None


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    active_model: Optional[str] = None
    active_draft_model: Optional[str] = None
    active_image_model: Optional[str] = None
    speculation_mode: str = "none"
    has_mtp: bool = False
    num_draft_tokens: Optional[int] = None
    backend: str = "llm"
    is_loading: bool = False
    loading_model: Optional[str] = None
    status: str = "idle"
