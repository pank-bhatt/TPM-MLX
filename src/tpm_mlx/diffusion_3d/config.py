"""Configuration models and default hyperparameters for MLX 3D diffusion."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ImageEncoderConfig:
    type: str = "dinov2-large"
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    patch_size: int = 14
    image_size: int = 518
    use_cls_token: bool = True
    mlp_ratio: float = 4.0
    qkv_bias: bool = True


@dataclass
class ShapeDiTConfig:
    hidden_size: int = 2048
    depth: int = 21
    num_heads: int = 16
    in_channels: int = 64
    context_dim: int = 1024
    num_latents: int = 4096
    num_moe_layers: int = 6
    num_experts: int = 8
    moe_top_k: int = 2
    qk_norm: bool = True
    qkv_bias: bool = False
    use_pos_emb: bool = False
    use_attention_pooling: bool = False
    bits: Optional[int] = None
    group_size: int = 64


@dataclass
class ShapeVAEConfig:
    num_latents: int = 4096
    embed_dim: int = 64
    width: int = 1024
    heads: int = 16
    num_decoder_layers: int = 16
    num_freqs: int = 8
    include_pi: bool = False
    qkv_bias: bool = False
    qk_norm: bool = True
    geo_decoder_downsample_ratio: int = 1
    geo_decoder_mlp_expand_ratio: int = 4
    geo_decoder_ln_post: bool = True
    scale_factor: float = 1.0
    label_type: str = "binary"


@dataclass
class Hunyuan3DConfig:
    model_type: str = "hunyuan3d-2.1"
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    dit: ShapeDiTConfig = field(default_factory=ShapeDiTConfig)
    vae: ShapeVAEConfig = field(default_factory=ShapeVAEConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hunyuan3DConfig":
        enc_cfg = data.get("image_encoder", {})
        dit_cfg = data.get("dit", {})
        vae_cfg = data.get("vae", {})
        return cls(
            model_type=data.get("model_type", "hunyuan3d-2.1"),
            image_encoder=ImageEncoderConfig(**enc_cfg) if isinstance(enc_cfg, dict) else enc_cfg,
            dit=ShapeDiTConfig(**dit_cfg) if isinstance(dit_cfg, dict) else dit_cfg,
            vae=ShapeVAEConfig(**vae_cfg) if isinstance(vae_cfg, dict) else vae_cfg,
        )
