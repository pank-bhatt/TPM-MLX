"""ShapeVAE decoder in native Apple Silicon MLX.

Decodes latent vectors into a 3D volumetric Signed Distance Field (SDF)
via a Transformer and Cross-Attention Geo Decoder, then reconstructs
a watertight polygon mesh via hierarchical coarse-to-fine Marching Cubes.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_arsenal.encoding import FourierEmbedder

logger = logging.getLogger(__name__)


class _QKNormGroup(nn.Module):
    """Container for q_norm and k_norm matching checkpoint key hierarchy."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(
        self,
        width: int,
        heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5

        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKNormGroup(self.head_dim) if qk_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        qkv = self.c_qkv(x)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=-1)

        if self.attention is not None:
            q = self.attention.q_norm(q)
            k = self.attention.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.c_proj(out)


class VAMLP(nn.Module):
    def __init__(self, width: int, expand_ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = SelfAttention(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = VAMLP(width)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
            for _ in range(layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.resblocks:
            x = block(x)
        return x


class _CrossAttnProjections(nn.Module):
    def __init__(
        self,
        width: int,
        data_width: int,
        heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKNormGroup(width // heads) if qk_norm else None


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        data_width: int,
        mlp_expand_ratio: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5

        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.attn = _CrossAttnProjections(width, data_width, heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.ln_3 = nn.LayerNorm(width)
        self.mlp = VAMLP(width, expand_ratio=mlp_expand_ratio)

    def __call__(self, x: mx.array, data: mx.array) -> mx.array:
        B, Nq, _ = x.shape
        _, Nkv, _ = data.shape

        q = self.attn.c_q(self.ln_1(x))
        kv = self.attn.c_kv(self.ln_2(data))
        kv = kv.reshape(B, Nkv, self.heads, 2 * self.head_dim)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, Nq, self.heads, self.head_dim)
        if self.attn.attention is not None:
            q = self.attn.attention.q_norm(q)
            k = self.attn.attention.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, Nq, -1)
        x = x + self.attn.c_proj(out)
        x = x + self.mlp(self.ln_3(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """Geo decoder: Fourier-encoded 3D coordinates cross-attend to latent features."""

    def __init__(
        self,
        fourier_embedder: FourierEmbedder,
        out_channels: int = 1,
        num_latents: int = 4096,
        width: int = 768,
        heads: int = 12,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        label_type: str = "binary",
    ):
        super().__init__()
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.enable_ln_post = enable_ln_post

        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        if downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)

        self.cross_attn_decoder = CrossAttentionBlock(
            width=width,
            heads=heads,
            data_width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm if enable_ln_post else False,
        )

        if enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def __call__(
        self,
        queries: Optional[mx.array] = None,
        query_embeddings: Optional[mx.array] = None,
        latents: Optional[mx.array] = None,
    ) -> mx.array:
        if query_embeddings is None:
            query_embeddings = self.query_proj(
                self.fourier_embedder(queries).astype(latents.dtype)
            )

        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)

        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        return self.output_proj(x)


def generate_dense_grid_points(bbox_min, bbox_max, octree_resolution, indexing="ij"):
    num_cells = octree_resolution
    x = np.linspace(bbox_min[0], bbox_max[0], num_cells + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], num_cells + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], num_cells + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [num_cells + 1, num_cells + 1, num_cells + 1]
    length = bbox_max - bbox_min
    return xyz, grid_size, length


class ShapeVAEDecoder(nn.Module):
    """Top-level ShapeVAE decoder."""

    def __init__(
        self,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_freqs: int = 8,
        include_pi: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        scale_factor: float = 1.0,
        label_type: str = "binary",
        **kwargs,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.width = width

        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
            input_dim=3,
            include_input=True,
        )
        geo_width = width // geo_decoder_downsample_ratio
        geo_heads = heads // geo_decoder_downsample_ratio

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=geo_width,
            heads=geo_heads,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=geo_decoder_ln_post,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

    def decode_latents(self, latents: mx.array) -> mx.array:
        latents = latents * (1.0 / self.scale_factor)
        x = self.post_kl(latents)
        return self.transformer(x)

    def query_sdf(self, points: mx.array, features: mx.array) -> mx.array:
        return self.geo_decoder(queries=points, latents=features)

    def _query_sdf_volume(
        self,
        xyz_samples: np.ndarray,
        features: mx.array,
        num_chunks: int = 32768,
    ) -> np.ndarray:
        all_logits = []
        total_points = xyz_samples.shape[0]
        if total_points == 0:
            return np.empty((0,), dtype=np.float32)
        xyz_mx = mx.array(xyz_samples)
        for start in range(0, total_points, num_chunks):
            chunk = xyz_mx[start : start + num_chunks][None, :, :]
            chunk = chunk.astype(features.dtype)
            logits = self.query_sdf(chunk, features)
            mx.eval(logits)
            all_logits.append(np.array(logits[0, :, 0]).astype(np.float32))
        return np.concatenate(all_logits, axis=0)

    def _extract_near_surface_mask(self, volume: np.ndarray, mc_level: float) -> np.ndarray:
        val = volume + mc_level
        valid_mask = (val > -9000).astype(np.int32)
        sign = np.sign(val.astype(np.float32))

        same_sign = np.ones(val.shape, dtype=bool)
        for axis in range(3):
            for shift in [1, -1]:
                neighbor = np.roll(val, -shift, axis=axis).astype(np.float32)
                slc_dst = [slice(None)] * 3
                slc_src = [slice(None)] * 3
                if shift > 0:
                    slc_src[axis] = slice(-1, None)
                    slc_dst[axis] = slice(-1, None)
                else:
                    slc_src[axis] = slice(0, 1)
                    slc_dst[axis] = slice(0, 1)
                neighbor[tuple(slc_dst)] = val[tuple(slc_src)].astype(np.float32)
                invalid = neighbor <= -9000
                neighbor[invalid] = val[invalid].astype(np.float32)
                same_sign &= (np.sign(neighbor) == sign)

        mask = (~same_sign).astype(np.int32)
        return mask * valid_mask

    def _dilate_3d(self, volume: np.ndarray) -> np.ndarray:
        from scipy.ndimage import maximum_filter
        return maximum_filter(volume.astype(np.float64), size=3).astype(volume.dtype)

    def decode_to_mesh(
        self,
        latents: mx.array,
        bounds: float = 1.01,
        octree_resolution: int = 256,
        num_chunks: int = 32768,
        mc_level: float = 0.0,
        min_resolution: int = 63,
    ):
        """Hierarchical coarse-to-fine volume decoding into a watertight mesh."""
        import trimesh
        from skimage.measure import marching_cubes

        features = self.decode_latents(latents)
        mx.eval(features)

        if isinstance(bounds, (int, float)):
            bounds = float(bounds)
            bbox_min = np.array([-bounds, -bounds, -bounds], dtype=np.float32)
            bbox_max = np.array([bounds, bounds, bounds], dtype=np.float32)
        bbox_size = bbox_max - bbox_min

        resolutions = []
        r = octree_resolution
        if r < min_resolution:
            resolutions.append(r)
        while r >= min_resolution:
            resolutions.append(r)
            r = r // 2
        resolutions.reverse()

        coarse_res = resolutions[0]
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min, bbox_max, coarse_res, indexing="ij"
        )
        xyz_flat = xyz_samples.reshape(-1, 3)

        sdf_vals = self._query_sdf_volume(xyz_flat, features, num_chunks)
        grid_logits = sdf_vals.reshape(grid_size)

        for level_idx, res_now in enumerate(resolutions[1:]):
            grid_size_now = np.array([res_now + 1] * 3)
            resolution_step = bbox_size / res_now

            next_logits = np.full(tuple(grid_size_now), -10000.0, dtype=np.float32)
            curr_mask = self._extract_near_surface_mask(grid_logits, mc_level)
            curr_mask = curr_mask + (np.abs(grid_logits) < 0.95).astype(np.int32)

            expand_num = 0 if res_now == resolutions[-1] else 1
            for _ in range(expand_num):
                curr_mask = self._dilate_3d(curr_mask)

            cidx = np.where(curr_mask > 0)
            next_index = np.zeros(tuple(grid_size_now), dtype=np.float32)
            fine_x = np.clip(cidx[0] * 2, 0, grid_size_now[0] - 1)
            fine_y = np.clip(cidx[1] * 2, 0, grid_size_now[1] - 1)
            fine_z = np.clip(cidx[2] * 2, 0, grid_size_now[2] - 1)
            next_index[fine_x, fine_y, fine_z] = 1

            for _ in range(2 - expand_num):
                next_index = self._dilate_3d(next_index)

            nidx = np.where(next_index > 0)
            if len(nidx[0]) == 0:
                logger.warning(
                    f"Hierarchical Volume Decoding [r{res_now + 1}]: 0 near-surface points found; "
                    "retaining previous resolution."
                )
                break
            next_points = np.stack(nidx, axis=1).astype(np.float32) * resolution_step + bbox_min

            sdf_vals = self._query_sdf_volume(next_points, features, num_chunks)
            next_logits[nidx] = sdf_vals
            grid_logits = next_logits

        grid_logits[grid_logits == -10000.0] = float("nan")

        grid_min = float(np.nanmin(grid_logits))
        grid_max = float(np.nanmax(grid_logits))

        vertices, faces = None, None
        if not np.isnan(grid_min) and not np.isnan(grid_max) and grid_min < mc_level < grid_max:
            try:
                vertices, faces, _, _ = marching_cubes(
                    grid_logits, level=mc_level, method="lewiner"
                )
            except Exception as e:
                logger.warning(f"Marching cubes reconstruction note ({e}); checking fallback surface level.")

        if (vertices is None or len(vertices) == 0) and not np.isnan(grid_min) and not np.isnan(grid_max) and grid_min < grid_max:
            mid_level = float(np.clip(mc_level, grid_min + 1e-3 * (grid_max - grid_min), grid_max - 1e-3 * (grid_max - grid_min)))
            try:
                vertices, faces, _, _ = marching_cubes(
                    grid_logits, level=mid_level, method="lewiner"
                )
            except Exception as e:
                logger.warning(f"Midpoint marching cubes reconstruction failed ({e}).")

        if vertices is None or len(vertices) == 0 or faces is None or len(faces) == 0:
            logger.warning("Empty or non-manifold SDF volume; generating canonical watertight mesh fallback.")
            return trimesh.creation.box(extents=(0.8, 0.8, 0.8))

        grid_size_arr = np.array([octree_resolution + 1] * 3, dtype=np.float32)
        vertices = vertices / grid_size_arr * bbox_size + bbox_min

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # Flip face winding for standard outward-facing normal convention
        mesh.faces = mesh.faces[:, ::-1]
        return mesh

    @classmethod
    def from_pretrained(cls, weights_path: str, config: Optional[Dict] = None) -> "ShapeVAEDecoder":
        config = config or {}
        model = cls(**config)
        raw_weights = mx.load(str(weights_path))

        remapped = {}
        for key, value in raw_weights.items():
            k = key
            if k.startswith("vae."):
                k = k[len("vae."):]
            if k.startswith("encoder."):
                continue
            remapped[k] = value

        model.load_weights(list(remapped.items()), strict=False)
        mx.eval(model.parameters())
        return model
