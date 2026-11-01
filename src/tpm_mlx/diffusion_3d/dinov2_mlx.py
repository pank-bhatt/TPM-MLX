"""DINOv2 image encoder in native Apple Silicon MLX.

Encodes an input reference image (518x518) into 1370 visual conditioning tokens
(37*37 spatial patch tokens + 1 CLS token) for the Hunyuan3D-DiT shape pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbeddings(nn.Module):
    """2D image patch embedding using Conv2d in channels-last layout."""

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Args: x: (B, H, W, C) -> Returns: (B, num_patches, embed_dim)."""
        x = self.projection(x)
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D)


class Embeddings(nn.Module):
    """DINOv2 embeddings: patch embeddings + CLS token + positional embeddings."""

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, in_channels, hidden_size
        )
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.mask_token = mx.zeros((1, hidden_size))
        self.position_embeddings = mx.zeros((1, num_patches + 1, hidden_size))
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        patches = self.patch_embeddings(x)
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        x = mx.concatenate([cls_tokens, patches], axis=1)
        x = x + self.position_embeddings
        return x


class DINOv2SelfAttention(nn.Module):
    """DINOv2 multi-head self-attention with separate Q, K, V projections."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return out.transpose(0, 2, 1, 3).reshape(B, N, C)


class DINOv2Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.attention = DINOv2SelfAttention(dim, num_heads, qkv_bias)
        self.output = nn.Linear(dim, dim)  # Matches HF output.dense

    def __call__(self, x: mx.array) -> mx.array:
        return self.output(self.attention(x))


class LayerScale(nn.Module):
    """Per-channel learnable scale factor."""

    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.lambda1


class DINOv2MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class DINOv2Layer(nn.Module):
    """Single transformer layer with pre-norm and LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = DINOv2Attention(dim, num_heads, qkv_bias)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv2MLP(dim, mlp_ratio)
        self.layer_scale2 = LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv2Encoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.layer = [
            DINOv2Layer(hidden_size, num_heads, mlp_ratio, qkv_bias)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.layer:
            x = blk(x)
        return x


class ImageEncoder(nn.Module):
    """DINOv2 ViT Image Encoder matching HuggingFace checkpoint structure.

    Maps 518x518 RGB images into (B, 1370, hidden_size) conditioning tokens.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        image_size: int = 518,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        patch_size: int = 14,
        use_cls_token: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.use_cls_token = use_cls_token
        self.num_patches = (image_size // patch_size) ** 2
        if use_cls_token:
            self.num_patches += 1

        self.embeddings = Embeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_size=hidden_size,
        )
        self.encoder = DINOv2Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

    def preprocess(self, image: mx.array) -> mx.array:
        """Normalizes (B, H, W, 3) in [0, 1] using standard ImageNet mean and std."""
        mean = mx.array(self.MEAN).reshape(1, 1, 1, 3).astype(image.dtype)
        std = mx.array(self.STD).reshape(1, 1, 1, 3).astype(image.dtype)
        return (image - mean) / std

    def __call__(self, image: mx.array, value_range: Optional[Tuple[float, float]] = (-1.0, 1.0)) -> mx.array:
        """Encodes image tensor in channels-last layout (B, H, W, 3).

        Args:
            image: (B, H, W, 3) input array.
            value_range: Value range of input (e.g. [-1, 1]) to normalize into [0, 1].

        Returns:
            (B, 1370, hidden_size) conditioning tokens.
        """
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = self.preprocess(image)
        x = self.embeddings(image)
        x = self.encoder(x)
        x = self.layernorm(x)

        if not self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def unconditional_embedding(self, batch_size: int = 1) -> mx.array:
        """Returns zero conditioning tokens for CFG."""
        return mx.zeros(
            (batch_size, self.num_patches, self.hidden_size),
            dtype=self.layernorm.weight.dtype,
        )

    @classmethod
    def from_pretrained(cls, weights_path: str, config: Optional[Dict] = None) -> "ImageEncoder":
        """Loads weights from safetensors with key remapping and Conv2d transposition."""
        config = config or {}
        model = cls(**config)
        raw_weights = mx.load(str(weights_path))

        remapped = {}
        for key, value in raw_weights.items():
            k = key
            if k.startswith("image_encoder."):
                k = k[len("image_encoder."):]

            # HuggingFace DINOv2 uses output.dense, MLX attention uses output
            k = k.replace(".attention.output.dense.", ".attention.output.")

            # PyTorch Conv2d (O, I, H, W) -> MLX Conv2d (O, H, W, I)
            if k == "embeddings.patch_embeddings.projection.weight" and value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)

            remapped[k] = value

        model.load_weights(list(remapped.items()))
        mx.eval(model.parameters())
        return model
