# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
Pre-allocated Key-Value Cache for Apple Silicon unified memory.
Eliminates dynamic memory allocations during autoregressive generation loops,
preventing unified memory spikes and Metal cache thrashing.
"""

from typing import Tuple
import mlx.core as mx
from mlx_lm.models.cache import KVCache


class PreAllocatedKVCache(KVCache):
    """
    A custom Key-Value cache that pre-allocates cache tensors up to max_size
    on the first update_and_fetch call to avoid dynamic memory allocation spikes.
    If the sequence length exceeds max_size, it falls back to standard dynamic growth.
    """
    def __init__(self, max_size: int = 4096):
        super().__init__()
        self.max_size = max_size

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        prev = self.offset

        # Pre-allocate key/value tensors on the first call when shape/dtype are known
        if self.keys is None:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            self.keys = mx.zeros((B, n_kv_heads, self.max_size, k_head_dim), dtype=keys.dtype)
            self.values = mx.zeros((B, n_kv_heads, self.max_size, v_head_dim), dtype=values.dtype)
            self.offset = 0
            prev = 0

        # Fallback to dynamic concatenation/growth if we exceed the pre-allocated max_size
        if (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            # Slice the existing pre-allocated arrays to current offset before concatenation
            if prev % self.step != 0:
                self.keys = self.keys[..., :prev, :]
                self.values = self.values[..., :prev, :]
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def is_trimmable(self) -> bool:
        """Returns True since PreAllocatedKVCache supports rollback via offset reduction."""
        return True

    def trim(self, n: int) -> int:
        """
        Trims the last n tokens from the cache by decreasing the offset pointer.
        Returns the actual number of tokens trimmed.
        """
        trimmed = min(self.offset, n)
        self.offset -= trimmed
        return trimmed
