# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import pytest
from tpm_mlx.engine import MLXEngine, PreAllocatedKVCache
from mlx_lm.generate import GenerationResponse
import mlx.core as mx


def test_reasoning_filter():
    """
    Directly tests the engine's reasoning filter state machine using mock data.
    """
    # Create engine instance with a dummy path (we won't call load)
    # To test the private _filter_reasoning, we don't need to load the model.
    # We can patch or just instantiate PreAllocatedKVCache/MLXEngine logic.
    
    # Let's mock a sequence of GenerationResponse outputs
    mock_responses = [
        GenerationResponse(text="Hello ", token=1, logprobs=mx.array([]), from_draft=False, prompt_tokens=1, prompt_tps=1.0, generation_tokens=1, generation_tps=1.0, peak_memory=1.0),
        GenerationResponse(text="world! <th", token=2, logprobs=mx.array([]), from_draft=False, prompt_tokens=1, prompt_tps=1.0, generation_tokens=2, generation_tps=1.0, peak_memory=1.0),
        GenerationResponse(text="ink> secret thoughts </th", token=3, logprobs=mx.array([]), from_draft=False, prompt_tokens=1, prompt_tps=1.0, generation_tokens=3, generation_tps=1.0, peak_memory=1.0),
        GenerationResponse(text="ink> assistant answer.", token=4, logprobs=mx.array([]), from_draft=False, prompt_tokens=1, prompt_tps=1.0, generation_tokens=4, generation_tps=1.0, peak_memory=1.0, finish_reason="stop")
    ]
    
    # We can create a dummy MLXEngine by bypassing constructor
    class DummyEngine(MLXEngine):
        def __init__(self):
            pass
            
    engine = DummyEngine()
    filtered = list(engine._filter_reasoning(mock_responses))
    
    # The output should be:
    # 1. "Hello "
    # 2. "world! " (the "<th" should be buffered, then "ink> secret thoughts </think>" is stripped, then " assistant answer.")
    # Let's concatenate all outputs
    output_text = "".join(r.text for r in filtered)
    assert "secret thoughts" not in output_text
    assert "<think>" not in output_text
    assert "</think>" not in output_text
    assert "Hello world!" in output_text
    assert "assistant answer." in output_text


def test_pre_allocated_cache():
    """Tests the PreAllocatedKVCache initialization and allocation logic."""
    cache = PreAllocatedKVCache(max_size=128)
    assert cache.keys is None
    assert cache.values is None
    
    # Mock keys/values to fetch
    # Shape: [batch, heads, seq_len, head_dim]
    k = mx.random.normal((1, 2, 4, 16))
    v = mx.random.normal((1, 2, 4, 16))
    
    k_fetched, v_fetched = cache.update_and_fetch(k, v)
    
    # After first call, keys/values should be allocated to max_size=128
    assert cache.keys is not None
    assert cache.keys.shape == (1, 2, 128, 16)
    assert cache.offset == 4
    
    # Fetched keys/values should be sliced up to offset
    assert k_fetched.shape == (1, 2, 4, 16)
    assert mx.allclose(k_fetched, k)
    
    # Try next step
    k2 = mx.random.normal((1, 2, 1, 16))
    v2 = mx.random.normal((1, 2, 1, 16))
    k_fetched2, v_fetched2 = cache.update_and_fetch(k2, v2)
    
    assert cache.offset == 5
    assert k_fetched2.shape == (1, 2, 5, 16)
    
    # Verify values are correctly set in the pre-allocated buffer
    assert mx.allclose(cache.keys[..., 4:5, :], k2)

    # Test trim & trimmable support
    assert cache.is_trimmable() is True
    trimmed = cache.trim(2)
    assert trimmed == 2
    assert cache.offset == 3

    # Test trimming beyond offset
    trimmed_excess = cache.trim(10)
    assert trimmed_excess == 3
    assert cache.offset == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required for GPU testing")
def test_gemma_model_loading():
    """
    Loads and runs gemma-4-e2b-it-4bit to verify engine load, weight patching, and generation.
    """
    # Use gemma-4-e2b-it-4bit which is cached
    model_id = "mlx-community/gemma-4-e2b-it-4bit"
    engine = MLXEngine(model_path_or_id=model_id, max_kv_size=128)
    
    assert engine.model is not None
    assert engine.tokenizer is not None
    
    # Simple generation test
    prompt = "Hello, my name is"
    stream = list(engine.generate_stream(prompt, max_tokens=10, temperature=0.0))
    
    assert len(stream) > 0
    generated_text = "".join(r.text for r in stream)
    assert len(generated_text) > 0
    
    # Check metrics are populated at the end
    last_resp = stream[-1]
    assert last_resp.generation_tokens > 0
    assert last_resp.generation_tps > 0.0
    assert last_resp.prompt_tps > 0.0
    
    # Check speculation properties for base Gemma model (no built-in MTP)
    assert engine.has_mtp is False
    assert engine.speculation_mode == "none"


def test_engine_speculation_properties():
    """Tests MLXEngine speculation mode and depth resolution."""
    class MockEngine(MLXEngine):
        def __init__(self, has_mtp=False, has_draft=False, user_depth=None):
            self.mtp_head = None
            if has_mtp:
                class DummyMTPHead:
                    depth = 3
                self.mtp_head = DummyMTPHead()
            self.draft_model = object() if has_draft else None
            self.user_num_draft_tokens = user_depth

    # Test MTP mode
    eng_mtp = MockEngine(has_mtp=True)
    assert eng_mtp.has_mtp is True
    assert eng_mtp.speculation_mode == "mtp"
    assert eng_mtp.num_draft_tokens == 3

    # Test Draft mode
    eng_draft = MockEngine(has_mtp=False, has_draft=True)
    assert eng_draft.has_mtp is False
    assert eng_draft.speculation_mode == "draft"
    assert eng_draft.num_draft_tokens == 2

    # Test user override
    eng_override = MockEngine(has_mtp=True, user_depth=5)
    assert eng_override.num_draft_tokens == 5

    # Test none
    eng_none = MockEngine(has_mtp=False, has_draft=False)
    assert eng_none.has_mtp is False
    assert eng_none.speculation_mode == "none"


def test_server_models_endpoint():
    """Tests FastAPI /v1/models response schema including speculation fields."""
    from fastapi.testclient import TestClient
    from tpm_mlx import server

    client = TestClient(server.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)


def test_engine_backend_detection():
    """Tests that MLXEngine._detect_backend correctly identifies LLM vs VLM models."""
    class DummyEngine(MLXEngine):
        def __init__(self):
            pass

    dummy = DummyEngine()
    
    # VLMs
    assert dummy._detect_backend({"model_type": "gemma4"}) == "vlm"
    assert dummy._detect_backend({"model_type": "qwen2_vl"}) == "vlm"
    assert dummy._detect_backend({"architectures": ["LlavaForConditionalGeneration"]}) == "vlm"
    assert dummy._detect_backend({"vision_config": {"depth": 10}}) == "vlm"
    
    # Pure LLMs
    assert dummy._detect_backend({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}) == "llm"
    assert dummy._detect_backend({"model_type": "mistral", "architectures": ["MistralForCausalLM"]}) == "llm"
    assert dummy._detect_backend({"model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"]}) == "llm"
