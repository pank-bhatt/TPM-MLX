"""HunYuanDiTPlain shape transformer in native Apple Silicon MLX.

21-block Flow-Matching Diffusion Transformer with U-Net skip connections,
standard MLPs on layers 0-14, and 8-expert MoE on layers 15-20.
"""

import math
import re
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_arsenal.moe import MoELayer


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, num_channels: int, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(half_dim).astype(mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar flow timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.time_embed = Timesteps(hidden_size)
        self.mlp_0 = nn.Linear(hidden_size, frequency_embedding_size)
        self.mlp_2 = nn.Linear(frequency_embedding_size, hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.time_embed(t).astype(self.mlp_0.weight.dtype)
        t = self.mlp_2(nn.gelu(self.mlp_0(t_freq)))
        return t[:, None, :]


class MLP(nn.Module):
    """Standard 2-layer MLP with GELU activation."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with optional QK RMSNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = mx.concatenate([q, k, v], axis=-1)
        qkv = qkv.reshape(B, N, self.num_heads, 3 * self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, N, -1))


class CrossAttention(nn.Module):
    """Multi-head cross-attention (latents attend to DINOv2 vision tokens)."""

    def __init__(
        self,
        qdim: int,
        kdim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.out_proj = nn.Linear(qdim, qdim)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        B, S1, _ = x.shape
        _, S2, _ = y.shape

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = mx.concatenate([k, v], axis=-1)
        kv = kv.reshape(B, S2, self.num_heads, 2 * self.head_dim)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, S1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S2, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S2, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, S1, -1))


class FinalLayer(nn.Module):
    """Final output projection."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm_final(x)
        x = x[:, 1:]  # Remove prepended conditioning token
        return self.linear(x)


class HunYuanDiTBlock(nn.Module):
    """Single DiT block with pre-norm, self-attention, cross-attention, and MLP or MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_dim: int = 1024,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        skip_connection: bool = False,
        timestep_modulate: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, context_dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)

        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.use_moe = use_moe
        if use_moe:
            def make_expert():
                return MLP(hidden_size)

            shared = MLP(hidden_size)
            self.moe = MoELayer(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=moe_top_k,
                expert_fn=make_expert,
                shared_expert=shared,
            )
        else:
            self.mlp = MLP(hidden_size)

        self.timestep_modulate = timestep_modulate
        if timestep_modulate:
            self.default_modulation_0 = nn.SiLU()
            self.default_modulation_1 = nn.Linear(hidden_size, hidden_size)

        if skip_connection:
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        else:
            self.skip_linear = None

    def __call__(
        self,
        x: mx.array,
        c: Optional[mx.array] = None,
        text_states: Optional[mx.array] = None,
        skip_value: Optional[mx.array] = None,
    ) -> mx.array:
        if self.skip_linear is not None and skip_value is not None:
            cat = mx.concatenate([skip_value, x], axis=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        if self.timestep_modulate and c is not None:
            shift = self.default_modulation_1(self.default_modulation_0(c))
            x = x + shift

        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), text_states)

        normed = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(normed)
        else:
            x = x + self.mlp(normed)

        return x


class HunYuanDiTPlain(nn.Module):
    """HunYuanDiT shape transformer with U-Net skip connections and MoE."""

    def __init__(
        self,
        input_size: int = 4096,
        in_channels: int = 64,
        hidden_size: int = 2048,
        context_dim: int = 1024,
        depth: int = 21,
        num_heads: int = 16,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_dim = context_dim

        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)

        self.blocks = [
            HunYuanDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_dim=context_dim,
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                skip_connection=(layer > depth // 2),
                timestep_modulate=False,
                use_moe=(depth - layer <= num_moe_layers),
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ]

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def __call__(self, x: mx.array, t: mx.array, contexts: Dict[str, Any], **kwargs) -> mx.array:
        """Forward pass.

        Args:
            x: (B, seq_len, in_channels) latent tokens (e.g. 1, 4096, 64).
            t: (B,) normalized timestep values in [0, 1].
            contexts: Dict containing 'main' -> DINOv2 tokens (B, 1370, 1024).

        Returns:
            (B, seq_len, in_channels) predicted flow velocity.
        """
        cond = contexts["main"]
        t_emb = self.t_embedder(t)
        x = self.x_embedder(x)

        # Prepend conditioning timestep token
        x = mx.concatenate([t_emb, x], axis=1)

        # U-Net skip loop
        skip_value_list = []
        for layer_idx, block in enumerate(self.blocks):
            skip_value = None if layer_idx <= self.depth // 2 else skip_value_list.pop()
            x = block(x, t_emb, cond, skip_value=skip_value)
            if layer_idx < self.depth // 2:
                skip_value_list.append(x)

        return self.final_layer(x)

    @classmethod
    def from_pretrained(cls, weights_path: str, config: Optional[Dict] = None) -> "HunYuanDiTPlain":
        """Loads safetensors weights with key remapping for MoE and MLP layers."""
        config = config or {}
        model = cls(**config)
        raw_weights = mx.load(str(weights_path))

        remapped = {}
        for key, value in raw_weights.items():
            k = key
            if k.startswith("dit."):
                k = k[len("dit."):]

            k = k.replace("t_embedder.mlp.0.", "t_embedder.mlp_0.")
            k = k.replace("t_embedder.mlp.2.", "t_embedder.mlp_2.")

            k = re.sub(r"(blocks\.\d+\.moe)\.gate\.weight$", r"\1.gate.gate.weight", k)
            k = re.sub(r"(blocks\.\d+\.moe\.experts\.\d+)\.net\.0\.proj\.", r"\1.fc1.", k)
            k = re.sub(r"(blocks\.\d+\.moe\.experts\.\d+)\.net\.2\.", r"\1.fc2.", k)
            k = re.sub(r"(blocks\.\d+\.moe)\.shared_experts\.net\.0\.proj\.", r"\1.shared_expert.fc1.", k)
            k = re.sub(r"(blocks\.\d+\.moe)\.shared_experts\.net\.2\.", r"\1.shared_expert.fc2.", k)

            remapped[k] = value

        # Detect quantization from weight keys (.scales present)
        quantized_keys = {k.rsplit(".scales", 1)[0] for k in remapped if k.endswith(".scales")}
        if quantized_keys:
            def _should_quantize(path, module):
                return isinstance(module, nn.Linear) and path in quantized_keys

            nn.quantize(
                model,
                group_size=config.get("group_size", 64),
                bits=config.get("bits", 8),
                class_predicate=_should_quantize,
            )

        model.load_weights(list(remapped.items()), strict=False)
        mx.eval(model.parameters())
        return model
