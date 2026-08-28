# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import pytest
import mlx.core as mx
import mlx.nn as nn

from tpm_mlx.mtp import MTPLayer, MTPHead
from tpm_mlx.mtp_weights import (
    detect_mtp_family,
    extract_and_remap_mtp_weights,
)
from tpm_mlx.speculation import SpeculationStats, mtp_generate_step
from tpm_mlx.engine import PreAllocatedKVCache


def test_mtp_layer_forward():
    """Tests MTPLayer forward pass and output shape."""
    hidden_size = 64
    layer = MTPLayer(hidden_size=hidden_size, rms_norm_eps=1e-6)
    
    # Mock hidden state [B=1, S=1, D=64] and token embedding [B=1, S=1, D=64]
    h = mx.random.normal((1, 1, hidden_size))
    emb = mx.random.normal((1, 1, hidden_size))
    
    out = layer(h, emb)
    assert out.shape == (1, 1, hidden_size)


def test_mtp_head_weight_sharing_and_draft():
    """Tests MTPHead drafting sequence with shared embeddings and LM head."""
    vocab_size = 100
    hidden_size = 32
    depth = 3
    
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    mtp_head = MTPHead(
        hidden_size=hidden_size,
        depth=depth,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
    )
    
    assert mtp_head.depth == 3
    assert len(mtp_head.layers) == 3
    
    # Initial token and hidden state
    tok = mx.array([42], mx.uint32)
    h = embed_tokens(tok[None])
    
    # Test single draft step
    draft_tok, draft_lp, next_h = mtp_head.draft_step(h, tok, layer_idx=0)
    assert draft_tok.shape == (1,)
    assert draft_lp.shape == (1, vocab_size)
    assert next_h.shape == (1, 1, hidden_size)
    
    # Test draft sequence (drafting 3 tokens)
    draft_tokens, draft_logprobs, final_h = mtp_head.draft_sequence(h, tok, num_draft_tokens=3)
    assert len(draft_tokens) == 3
    assert len(draft_logprobs) == 3
    assert final_h.shape == (1, 1, hidden_size)


def test_detect_mtp_family():
    """Tests MTP architecture family detection from weight key patterns."""
    qwen_weights = {
        "model.embed_tokens.weight": mx.zeros((10, 10)),
        "mtp.layers.0.enorm.weight": mx.zeros((10,)),
        "mtp.layers.0.proj.weight": mx.zeros((10, 20)),
    }
    assert detect_mtp_family(qwen_weights) == "qwen3"
    
    mimo_weights = {
        "model.embed_tokens.weight": mx.zeros((10, 10)),
        "model.mtp_layers.0.enorm.weight": mx.zeros((10,)),
        "model.mtp_layers.0.input_proj.weight": mx.zeros((10, 20)),
    }
    assert detect_mtp_family(mimo_weights) == "mimo"
    
    deepseek_weights = {
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
        "model.layers.61.enorm.weight": mx.zeros((10,)),
    }
    assert detect_mtp_family(deepseek_weights) == "deepseek_v3"
    
    standard_weights = {
        "model.embed_tokens.weight": mx.zeros((10, 10)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((10, 10)),
    }
    assert detect_mtp_family(standard_weights) is None


def test_extract_and_remap_mtp_weights():
    """Tests remapping checkpoint weights to generic MTPHead parameter paths."""
    qwen_weights = {
        "mtp.layers.0.enorm.weight": mx.ones((32,)),
        "mtp.layers.0.hnorm.weight": mx.ones((32,)),
        "mtp.layers.0.eh_proj.weight": mx.ones((32, 64)),
        "mtp.layers.0.norm.weight": mx.ones((32,)),
        "mtp.layers.1.enorm.weight": mx.ones((32,)),
        "mtp.layers.1.hnorm.weight": mx.ones((32,)),
        "mtp.layers.1.eh_proj.weight": mx.ones((32, 64)),
        "mtp.layers.1.norm.weight": mx.ones((32,)),
    }
    
    remapped, depth = extract_and_remap_mtp_weights(qwen_weights, family="qwen3", hidden_size=32)
    assert depth == 2
    assert "layers.0.enorm.weight" in remapped
    assert "layers.0.hnorm.weight" in remapped
    assert "layers.0.proj.weight" in remapped
    assert "layers.0.norm.weight" in remapped
    assert "layers.1.enorm.weight" in remapped
    assert "layers.1.proj.weight" in remapped


def test_speculation_stats():
    """Tests telemetry data calculation in SpeculationStats."""
    stats = SpeculationStats()
    assert stats.acceptance_rate == 0.0
    assert stats.avg_tokens_per_step == 1.0
    
    stats.speculation_steps = 10
    stats.draft_tokens_total = 20
    stats.accepted_tokens_total = 15
    
    assert stats.acceptance_rate == 0.75  # 15 / 20
    assert stats.avg_tokens_per_step == 2.5  # (15 + 10) / 10


def test_mtp_generate_step_mock():
    """Tests the mtp_generate_step generation loop with a small mock model."""
    vocab_size = 50
    hidden_size = 16
    
    class MockTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            
        def __call__(self, x, cache=None):
            h = self.embed_tokens(x)
            if cache is not None and len(cache) > 0 and cache[0] is not None:
                B, S, D = h.shape
                k = h.reshape(B, 1, S, D)
                v = h.reshape(B, 1, S, D)
                cache[0].update_and_fetch(k, v)
            return h

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockTrunk()
            self.embed_tokens = self.model.embed_tokens
            self.layers = [nn.Identity()]
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
        def __call__(self, x, cache=None):
            h = self.model(x, cache=cache)
            return self.lm_head(h)
            
    mock_model = MockModel()
    mtp_head = MTPHead(
        hidden_size=hidden_size,
        depth=2,
        embed_tokens=mock_model.embed_tokens,
        lm_head=mock_model.lm_head,
    )
    
    prompt = mx.array([1, 2, 3], mx.uint32)
    stats = SpeculationStats()
    
    # Run mock generation for 6 tokens
    generator = mtp_generate_step(
        prompt=prompt,
        model=mock_model,
        mtp_head=mtp_head,
        num_draft_tokens=2,
        max_tokens=6,
        stats=stats,
    )
    
    results = list(generator)
    assert len(results) == 6
    assert stats.speculation_steps > 0
    # Every yielded item should be (token_id, logprobs, from_draft)
    for tok, lp, from_draft in results:
        assert isinstance(tok, int)
        assert isinstance(from_draft, bool)
        assert lp.shape == (vocab_size,)
