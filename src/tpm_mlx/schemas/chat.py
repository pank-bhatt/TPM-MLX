# Copyright © 2026 TPM-MLX Authors. All rights reserved.

from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    reasoning: Optional[bool] = Field(default=None, description="Toggles outputting reasoning <think> blocks")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="OpenAI-compatible tools list")

