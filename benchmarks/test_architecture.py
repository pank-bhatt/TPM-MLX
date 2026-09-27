# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
Automated unit and integration tests for the decoupled clean architecture:
Schemas, Router endpoints, Director prompt distillation, Static asset serving,
and PreAllocatedKVCache memory management.
"""

import pytest
from fastapi.testclient import TestClient
import mlx.core as mx

from tpm_mlx.server import app
from tpm_mlx.cache import PreAllocatedKVCache
from tpm_mlx.vlm import extract_text_and_images
from tpm_mlx.director import distill_visual_prompt
from tpm_mlx.schemas import (
    ChatMessage,
    ChatCompletionRequest,
    LoadModelRequest,
    ImageGenerationApiRequest,
    ImageEditApiRequest,
)


def test_schema_validations():
    """Validates Pydantic schema validation and defaults."""
    # Chat message
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    # Multimodal message
    multi_msg = ChatMessage(role="user", content=[{"type": "text", "text": "Hi"}])
    assert len(multi_msg.content) == 1

    # Chat completion request
    req = ChatCompletionRequest(model="test-model", messages=[msg])
    assert req.max_tokens == 4096
    assert req.temperature == 0.0
    assert req.stream is False
    assert req.reasoning is None

    # Image gen request
    img_req = ImageGenerationApiRequest(prompt="A landscape")
    assert img_req.size == "1024x1024"
    assert img_req.steps == 4
    assert img_req.response_format == "url"

    # Image edit request
    edit_req = ImageEditApiRequest(image="/images/test.png", prompt="Make it sunny")
    assert edit_req.guidance == 2.5
    assert edit_req.steps == 4


def test_vlm_extract_text_and_images():
    """Validates multimodal vision payload extraction."""
    # 1. Plain text messages
    msgs = [ChatMessage(role="user", content="Hello world")]
    formatted, imgs = extract_text_and_images(msgs)
    assert len(formatted) == 1
    assert formatted[0]["content"] == "Hello world"
    assert len(imgs) == 0

    # 2. Multimodal list message without valid image
    msgs2 = [ChatMessage(role="user", content=[
        {"type": "text", "text": "Describe this:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,invalid"}}
    ])]
    formatted2, imgs2 = extract_text_and_images(msgs2)
    assert len(formatted2) == 1
    assert formatted2[0]["content"] == "Describe this:"


def test_preallocated_kv_cache():
    """Validates PreAllocatedKVCache trimming and allocation behavior."""
    cache = PreAllocatedKVCache(max_size=64)
    assert cache.is_trimmable() is True

    # Dummy keys and values
    keys = mx.zeros((1, 4, 10, 32))
    values = mx.zeros((1, 4, 10, 32))

    k, v = cache.update_and_fetch(keys, values)
    assert cache.offset == 10
    assert k.shape == (1, 4, 10, 32)

    # Trim 4 tokens
    trimmed = cache.trim(4)
    assert trimmed == 4
    assert cache.offset == 6


def test_static_modular_assets_serving():
    """Verifies that all modular CSS and JS static assets are served with HTTP 200."""
    client = TestClient(app)

    # Root HTML
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "/static/css/style.css" in r_root.text
    assert "/static/js/app.js" in r_root.text

    # CSS modules
    for css_file in ("style.css", "theme.css", "layout.css", "components.css"):
        resp = client.get(f"/static/css/{css_file}")
        assert resp.status_code == 200, f"Failed to serve {css_file}"
        assert len(resp.text) > 0

    # JS modules
    for js_file in ("app.js", "theme.js", "markdown.js", "lightbox.js", "sessions.js", "api.js"):
        resp = client.get(f"/static/js/{js_file}")
        assert resp.status_code == 200, f"Failed to serve {js_file}"
        assert len(resp.text) > 0


def test_director_fallback_when_no_engine():
    """Verifies AI Visual Director returns original prompt safely when engine is None."""
    import asyncio
    prompt = "A red sports car"
    result = asyncio.run(distill_visual_prompt(prompt, context=None, is_edit=False))
    assert result == prompt
