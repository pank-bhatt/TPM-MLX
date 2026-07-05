# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import time
import json
import copy
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import _download, load_model, load_tokenizer
from mlx_lm.models.cache import KVCache
from mlx_lm.generate import stream_generate, GenerationResponse


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


class MLXEngine:
    """
    TPM-MLX optimized inference engine wrapping mlx-lm with custom patching,
    pre-allocated KV caching, zero CPU-GPU sync logic, and reasoning filtering.
    """
    def __init__(self, model_path_or_id: str, max_kv_size: int = 4096):
        self.model_path_or_id = model_path_or_id
        self.max_kv_size = max_kv_size
        
        # Remap newer unified Gemma 4 architectures to the supported gemma4 implementation
        from mlx_lm.utils import MODEL_REMAPPING
        MODEL_REMAPPING["gemma4_unified"] = "gemma4"
        MODEL_REMAPPING["gemma4_unified_assistant"] = "gemma4"
        
        # Download and cache model path first
        self.model_path = Path(_download(model_path_or_id))
        
        # Resolve config and apply dynamic patches
        self.config = self._load_and_patch_config()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_and_patch_config(self) -> Dict[str, Any]:
        """
        Loads the model config.json and dynamically overrides specific keys
        to fix sliding window / KeyError exceptions in gemma4_assistant models.
        """
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Gemma 4 Assistant override for sliding_attention KeyErrors
        # Overrides num_kv_shared_layers to 0 to bypass sliding_attention checks
        if config.get("model_type") == "gemma4_assistant":
            if "text_config" in config and isinstance(config["text_config"], dict):
                if config["text_config"].get("num_kv_shared_layers", 0) > 0:
                    config["text_config"]["num_kv_shared_layers"] = 0
                    
        return config

    def _load_model_and_tokenizer(self):
        """
        Loads the tokenizer and model. Uses strict=False to bypass key mismatches.
        """
        # Load model using strict=False to skip missing weights
        model, _ = load_model(
            self.model_path, 
            lazy=False, 
            strict=False, 
            model_config=self.config
        )
        
        # Load corresponding tokenizer
        tokenizer = load_tokenizer(
            self.model_path, 
            tokenizer_config_extra=None, 
            eos_token_ids=self.config.get("eos_token_id", None)
        )
        
        return model, tokenizer

    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 4096, 
        temperature: float = 0.0,
        show_reasoning: bool = False,
    ) -> Generator[GenerationResponse, None, None]:
        """
        A streaming generator wrapping stream_generate with PreAllocatedKVCache
        and reasoning filtering.
        """
        # Construct pre-allocated prompt cache, preserving hybrid structures (e.g. RotatingKVCache in Gemma 4)
        from mlx_lm.models import cache as mlx_cache
        prompt_cache = mlx_cache.make_prompt_cache(self.model, max_kv_size=self.max_kv_size)
        for i, c in enumerate(prompt_cache):
            if type(c) is KVCache:
                prompt_cache[i] = PreAllocatedKVCache(max_size=self.max_kv_size)
        
        # Build generation keyword arguments
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)
        
        gen_kwargs = {
            "max_tokens": max_tokens,
            "sampler": sampler,
            "prompt_cache": prompt_cache,
        }
        
        # Call underlying optimized stream_generate
        raw_stream = stream_generate(self.model, self.tokenizer, prompt, **gen_kwargs)
        
        # Wrap stream with reasoning filter if show_reasoning is False
        if show_reasoning:
            yield from self._normalize_reasoning(raw_stream)
        else:
            yield from self._filter_reasoning(raw_stream)

    def _normalize_reasoning(
        self, 
        raw_stream: Generator[GenerationResponse, None, None]
    ) -> Generator[GenerationResponse, None, None]:
        """
        Normalizes reasoning tags to <think>...</think> on the fly so clients
        that expect standard tags don't break.
        """
        inside_think = False
        buffer = ""
        
        start_tags = ["<think>", "<|channel>thought"]
        active_end_tag = None
        
        for response in raw_stream:
            buffer += response.text
            yield_text = ""
            
            while True:
                if not inside_think:
                    # Look for any of the start tags
                    found_tag = None
                    found_idx = -1
                    for tag in start_tags:
                        idx = buffer.find(tag)
                        if idx != -1:
                            if found_idx == -1 or idx < found_idx:
                                found_idx = idx
                                found_tag = tag
                    
                    if found_tag is not None:
                        # Capture text before the tag
                        prefix = buffer[:found_idx]
                        if prefix:
                            yield_text += prefix
                        # Output normalized start tag
                        yield_text += "<think>\n" if found_tag != "<think>" else "<think>"
                        # Strip original start tag and transition
                        buffer = buffer[found_idx + len(found_tag):]
                        inside_think = True
                        active_end_tag = "</think>" if found_tag == "<think>" else "<channel|>"
                    else:
                        # Check if buffer ends with a partial start tag
                        keep_len = 0
                        for tag in start_tags:
                            for i in range(1, len(tag)):
                                if buffer.endswith(tag[:i]):
                                    keep_len = max(keep_len, i)
                                    break
                        
                        if keep_len > 0:
                            yield_text += buffer[:-keep_len]
                            buffer = buffer[-keep_len:]
                        else:
                            yield_text += buffer
                            buffer = ""
                        break
                else:
                    # Look for active end tag
                    idx = buffer.find(active_end_tag)
                    if idx != -1:
                        # Capture text inside the think block
                        yield_text += buffer[:idx]
                        # Output normalized end tag
                        yield_text += "\n</think>\n" if active_end_tag != "</think>" else "</think>"
                        # Strip original end tag and transition back
                        buffer = buffer[idx + len(active_end_tag):]
                        inside_think = False
                        active_end_tag = None
                    else:
                        # We are inside a thinking block, so yield everything except partial end tags
                        keep_len = 0
                        for i in range(1, len(active_end_tag)):
                            if buffer.endswith(active_end_tag[:i]):
                                keep_len = i
                                break
                        
                        if keep_len > 0:
                            yield_text += buffer[:-keep_len]
                            buffer = buffer[-keep_len:]
                        else:
                            yield_text += buffer
                            buffer = ""
                        break
            
            # If we have clean text to yield, construct response
            if yield_text:
                resp_copy = copy.copy(response)
                resp_copy.text = yield_text
                yield resp_copy
            elif response.finish_reason is not None:
                # Always yield final response to communicate finish_reason and final stats
                resp_copy = copy.copy(response)
                resp_copy.text = ""
                yield resp_copy

    def _filter_reasoning(
        self, 
        raw_stream: Generator[GenerationResponse, None, None]
    ) -> Generator[GenerationResponse, None, None]:
        """
        State machine to filter out reasoning tags and all content inside them
        from the output stream. It supports both DeepSeek-style (<think>...</think>)
        and Gemma4-style (<|channel>thought...<channel|>) tags dynamically.
        """
        inside_think = False
        buffer = ""
        
        start_tags = ["<think>", "<|channel>thought"]
        active_end_tag = None
        
        for response in raw_stream:
            buffer += response.text
            yield_text = ""
            
            while True:
                if not inside_think:
                    # Look for any of the start tags
                    found_tag = None
                    found_idx = -1
                    for tag in start_tags:
                        idx = buffer.find(tag)
                        if idx != -1:
                            if found_idx == -1 or idx < found_idx:
                                found_idx = idx
                                found_tag = tag
                    
                    if found_tag is not None:
                        # Capture text before the tag
                        prefix = buffer[:found_idx]
                        if prefix:
                            yield_text += prefix
                        # Strip start tag and transition
                        buffer = buffer[found_idx + len(found_tag):]
                        inside_think = True
                        active_end_tag = "</think>" if found_tag == "<think>" else "<channel|>"
                    else:
                        # Check if buffer ends with a partial start tag
                        keep_len = 0
                        for tag in start_tags:
                            for i in range(1, len(tag)):
                                if buffer.endswith(tag[:i]):
                                    keep_len = max(keep_len, i)
                                    break
                        
                        if keep_len > 0:
                            yield_text += buffer[:-keep_len]
                            buffer = buffer[-keep_len:]
                        else:
                            yield_text += buffer
                            buffer = ""
                        break
                else:
                    # Look for active end tag
                    idx = buffer.find(active_end_tag)
                    if idx != -1:
                        # Strip end tag and transition back
                        buffer = buffer[idx + len(active_end_tag):]
                        inside_think = False
                        active_end_tag = None
                    else:
                        # Discard buffer content since we are inside a thinking block.
                        # Check if buffer ends with a partial end tag
                        keep_len = 0
                        for i in range(1, len(active_end_tag)):
                            if buffer.endswith(active_end_tag[:i]):
                                keep_len = i
                                break
                        
                        if keep_len > 0:
                            buffer = buffer[-keep_len:]
                        else:
                            buffer = ""
                        break
            
            # If we have clean text to yield, construct response
            if yield_text:
                resp_copy = copy.copy(response)
                resp_copy.text = yield_text
                yield resp_copy
            elif response.finish_reason is not None:
                # Always yield final response to communicate finish_reason and final stats
                resp_copy = copy.copy(response)
                resp_copy.text = ""
                yield resp_copy
