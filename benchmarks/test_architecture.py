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


def test_models_image_draft_isolation():
    """Verifies that image models report zero draft/mtp parameters and no draft assistants are active."""
    from tpm_mlx import state
    client = TestClient(app)

    prev_model = state.loaded_model_id
    prev_draft = state.loaded_draft_model_id
    try:
        state.loaded_model_id = "justintime47/Z-Image-Turbo-MLX-Serve-8bit"
        # Even if a stale draft was left in memory, list_models should isolate it
        state.loaded_draft_model_id = "mlx-community/gemma-4-E2B-it-assistant-bf16"

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "justintime47/Z-Image-Turbo-MLX-Serve-8bit"
        assert data["active_draft_model"] is None
        assert data["has_mtp"] is False
        assert data["speculation_mode"] == "none"
        assert data["num_draft_tokens"] == 0
        assert data["backend"] == "image"

        # Check in the data item list
        active_items = [m for m in data["data"] if m["id"] == "justintime47/Z-Image-Turbo-MLX-Serve-8bit"]
        assert len(active_items) == 1
        item = active_items[0]
        assert item["is_image"] is True
        assert item["is_draft"] is False
        assert item["draft_model"] is None
        assert item["has_mtp"] is False

        # Strictly verify that NO draft assistant has active: True
        active_drafts = [m for m in data["data"] if m.get("is_draft") and m.get("active")]
        assert len(active_drafts) == 0

        # Strictly verify that no OTHER image model is marked active
        other_active_images = [m for m in data["data"] if m.get("is_image") and m["id"] != "justintime47/Z-Image-Turbo-MLX-Serve-8bit" and m.get("active")]
        assert len(other_active_images) == 0
    finally:
        state.loaded_model_id = prev_model
        state.loaded_draft_model_id = prev_draft


def test_models_text_draft_active():
    """Verifies that when a text LLM is active, image models in catalog have active: False."""
    from tpm_mlx import state
    client = TestClient(app)

    prev_model = state.loaded_model_id
    prev_draft = state.loaded_draft_model_id
    try:
        state.loaded_model_id = "mlx-community/gemma-4-e2b-it-4bit"
        state.loaded_draft_model_id = "mlx-community/gemma-4-E2B-it-assistant-bf16"

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_model"] == "mlx-community/gemma-4-e2b-it-4bit"
        assert data["active_draft_model"] == "mlx-community/gemma-4-E2B-it-assistant-bf16"

        # Check that image models in cache are NOT active
        active_images = [m for m in data["data"] if m.get("is_image") and m.get("active")]
        assert len(active_images) == 0

        # Exactly one LLM base model is active
        active_llms = [m for m in data["data"] if not m.get("is_draft") and not m.get("is_image") and m.get("active")]
        assert len(active_llms) == 1
        assert active_llms[0]["id"] == "mlx-community/gemma-4-e2b-it-4bit"
    finally:
        state.loaded_model_id = prev_model
        state.loaded_draft_model_id = prev_draft


def test_models_loading_telemetry_and_alias():
    """Verifies that /v1/models reports in-flight loading state and /v1/models/load is aliased."""
    from tpm_mlx import state
    client = TestClient(app)

    prev_loading = state.is_loading
    prev_loading_model = state.loading_model_id
    try:
        state.is_loading = True
        state.loading_model_id = "mlx-community/Qwen3.8-27B-4bit"

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_loading"] is True
        assert data["loading_model"] == "mlx-community/Qwen3.8-27B-4bit"
        assert data["status"] == "loading"

        # Check in the data item list
        loading_items = [m for m in data["data"] if m["id"] == "mlx-community/Qwen3.8-27B-4bit"]
        assert len(loading_items) >= 1
        assert loading_items[0]["status"] == "loading"
    finally:
        state.is_loading = prev_loading
        state.loading_model_id = prev_loading_model

    # Verify unloading works via /v1/models/load alias
    resp_unload = client.post("/v1/models/load", json={"model": "none"})
    assert resp_unload.status_code == 200
    assert resp_unload.json()["status"] == "success"


def test_chat_completions_503_during_model_loading():
    """Verifies that chat completions returns 503 with Retry-After when model is actively loading."""
    from tpm_mlx import state
    client = TestClient(app)

    prev_engine = state.engine
    prev_loading = state.is_loading
    prev_loading_model = state.loading_model_id
    try:
        state.engine = None
        state.is_loading = True
        state.loading_model_id = "mlx-community/Qwen3.8-27B-4bit"
        state.loading_task = None  # No active task to wait on, triggers immediate 503

        resp = client.post("/v1/chat/completions", json={
            "model": "mlx-community/Qwen3.8-27B-4bit",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert resp.status_code == 503
        assert "currently loading" in resp.json()["detail"]
        assert resp.headers.get("Retry-After") == "5"
    finally:
        state.engine = prev_engine
        state.is_loading = prev_loading
        state.loading_model_id = prev_loading_model


def test_load_engine_instant_return_if_already_loaded():
    """Verifies that load_engine returns immediately when requested model is already active in memory."""
    import asyncio
    from tpm_mlx import state
    from unittest.mock import MagicMock

    dummy_engine = MagicMock()
    dummy_engine.max_kv_size = 4096
    dummy_engine.speculation_mode = "none"

    prev_engine = state.engine
    prev_model = state.loaded_model_id
    prev_draft = state.loaded_draft_model_id
    try:
        state.engine = dummy_engine
        state.loaded_model_id = "test-model-active"
        state.loaded_draft_model_id = None

        res = asyncio.run(state.load_engine(
            model_id="test-model-active",
            max_kv_size=4096,
            draft_model=None,
        ))
        assert res is dummy_engine
    finally:
        state.engine = prev_engine
        state.loaded_model_id = prev_model
        state.loaded_draft_model_id = prev_draft


def test_auto_pair_companion_mtp_draft():
    """Verifies that __AUTO__ or unprovided draft model auto-pairs with companion MTP assistant."""
    from unittest.mock import patch, AsyncMock
    from tpm_mlx import state

    client = TestClient(app)
    mock_engine = AsyncMock()

    with patch("tpm_mlx.state.load_engine", new=mock_engine), \
         patch("tpm_mlx.routes.models.get_cached_models") as mock_cached:
        mock_cached.return_value = [
            {"repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit", "is_draft": False},
            {"repo_id": "mlx-community/Qwen3.6-35B-A3B-MTP-4bit", "is_draft": True},
        ]
        resp = client.post("/v1/models/load", json={
            "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "draft_model": "__AUTO__"
        })
        assert resp.status_code == 200
        # Verify load_engine was called with the auto-paired MTP draft model
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args.kwargs
        assert call_kwargs["draft_model"] == "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"




