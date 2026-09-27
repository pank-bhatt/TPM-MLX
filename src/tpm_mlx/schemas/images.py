# Copyright © 2026 TPM-MLX Authors. All rights reserved.

from typing import Optional
from pydantic import BaseModel, Field


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
