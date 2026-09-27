# Copyright © 2026 TPM-MLX Authors. All rights reserved.

from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    reasoning: Optional[bool] = Field(default=None, description="Toggles outputting reasoning <think> blocks")
