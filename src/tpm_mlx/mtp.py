# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, make_prompt_cache


class MTPLayer(nn.Module):
    """
    A single depth layer of Multi-Token Prediction (MTP).
    
    Standard MTP Layer Architecture (Shared across Qwen 3.x, Xiaomi MiMo, DeepSeek V3):
        1. e_norm = RMSNorm(embed_dim)
        2. h_norm = RMSNorm(hidden_dim)
        3. proj = Linear(embed_dim + hidden_dim -> hidden_dim)
        4. block = DecoderLayer / TransformerBlock
        5. norm = RMSNorm(hidden_dim)
    """
    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float = 1e-6,
        block: Optional[nn.Module] = None,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim or hidden_size
        
        self.enorm = nn.RMSNorm(self.embed_dim, eps=rms_norm_eps)
        self.hnorm = nn.RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.proj = nn.Linear(self.embed_dim + self.hidden_size, self.hidden_size, bias=False)
        self.block = block
        self.norm = nn.RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_state: mx.array,
        token_embedding: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """
        Forward pass for a single MTP layer.
        
        Args:
            hidden_state: [B, 1, hidden_dim] from trunk or preceding MTP layer
            token_embedding: [B, 1, embed_dim] embedding of candidate/current token
            mask: Optional causal attention mask
            cache: Optional KV cache for this layer's attention block
            
        Returns:
            out_hidden_state: [B, 1, hidden_dim] transformed hidden representation
        """
        e = self.enorm(token_embedding)
        h = self.hnorm(hidden_state)
        fused = self.proj(mx.concatenate([e, h], axis=-1))
        
        if self.block is not None:
            if mask is not None or cache is not None:
                h_out = self.block(fused, mask=mask, cache=cache)
            else:
                h_out = self.block(fused)
        else:
            h_out = fused
            
        return h_out


class MTPHead(nn.Module):
    """
    Unified MTP Head holding D sequential MTPLayer modules.
    
    Shares the base model's embed_tokens and lm_head to avoid parameter bloat.
    """
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        embed_tokens: nn.Module,
        lm_head: Union[nn.Linear, nn.Module, Callable[[mx.array], mx.array]],
        rms_norm_eps: float = 1e-6,
        embed_dim: Optional[int] = None,
        layers: Optional[List[MTPLayer]] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.embed_dim = embed_dim or hidden_size
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        
        if layers is not None:
            self.layers = layers
        else:
            self.layers = [
                MTPLayer(hidden_size=hidden_size, rms_norm_eps=rms_norm_eps, embed_dim=self.embed_dim)
                for _ in range(depth)
            ]

    def unembed(self, h: mx.array) -> mx.array:
        """Projects hidden state through the shared language model head."""
        if hasattr(self.lm_head, "as_linear"):
            return self.lm_head.as_linear(h)
        elif callable(self.lm_head):
            return self.lm_head(h)
        else:
            raise TypeError(f"Unsupported lm_head type: {type(self.lm_head)}")

    def draft_step(
        self,
        hidden_state: mx.array,
        token: mx.array,
        layer_idx: int = 0,
        cache: Optional[Any] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Runs one step of drafting through the MTP layer at index `layer_idx`.
        
        Args:
            hidden_state: [B, 1, hidden_dim]
            token: [B, 1] or [1] integer token array
            layer_idx: index in [0, depth - 1]
            cache: KV cache for the layer's attention block
            sampler: Optional token sampling function
            
        Returns:
            (draft_token, draft_logprobs, next_hidden_state)
        """
        if token.ndim == 1:
            token = token[None]
            
        # 1. Embed current token
        token_emb = self.embed_tokens(token)
        
        # 2. Forward through MTP layer
        layer = self.layers[layer_idx]
        next_h = layer(hidden_state, token_emb, cache=cache)
        
        # 3. Apply layer output norm & shared LM head
        normed_h = layer.norm(next_h)
        logits = self.unembed(normed_h)
        logits = logits[:, -1, :]
        
        # 4. Compute logprobs & sample
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if sampler is not None:
            draft_token = sampler(logprobs)
        else:
            draft_token = mx.argmax(logprobs, axis=-1)
            
        return draft_token, logprobs, next_h

    def draft_sequence(
        self,
        hidden_state: mx.array,
        initial_token: mx.array,
        num_draft_tokens: int,
        caches: Optional[List[Any]] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
    ) -> Tuple[List[mx.array], List[mx.array], mx.array]:
        """
        Sequentially drafts up to `num_draft_tokens` candidates.
        
        Returns:
            (draft_tokens_list, draft_logprobs_list, final_hidden_state)
        """
        draft_tokens = []
        draft_logprobs = []
        
        curr_h = hidden_state
        curr_tok = initial_token
        
        n_steps = min(num_draft_tokens, self.depth)
        
        for k in range(n_steps):
            cache_k = caches[k] if caches is not None and k < len(caches) else None
            tok, lp, next_h = self.draft_step(
                hidden_state=curr_h,
                token=curr_tok,
                layer_idx=k,
                cache=cache_k,
                sampler=sampler,
            )
            draft_tokens.append(tok)
            draft_logprobs.append(lp)
            curr_tok = tok
            curr_h = next_h
            
        return draft_tokens, draft_logprobs, curr_h
