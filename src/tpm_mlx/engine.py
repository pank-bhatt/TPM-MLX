# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import time
import json
import copy
import logging
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Tuple, Union, Callable

logger = logging.getLogger("tpm-mlx.engine")

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


class MLXEngine:
    """
    Unified High-Performance Engine for Apple Silicon (MLX).
    Auto-detects model type and routes to either:
      - 'llm': mlx-lm with native MTP trunk self-speculation and pre-allocated KV cache.
      - 'vlm': mlx-vlm for multimodal Vision-Language Models & Gemma 4 MTP assistant drafters.
    """
    def __init__(
        self, 
        model_path_or_id: str, 
        max_kv_size: int = 4096,
        draft_model_path_or_id: Optional[str] = None,
        enable_mtp: bool = True,
        num_draft_tokens: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        self.model_path_or_id = model_path_or_id
        self.max_kv_size = max_kv_size
        self.draft_model_path_or_id = draft_model_path_or_id
        self.enable_mtp = enable_mtp
        self.user_num_draft_tokens = num_draft_tokens
        
        # Remap newer unified Gemma 4 architectures if mlx-lm is used
        from mlx_lm.utils import MODEL_REMAPPING
        MODEL_REMAPPING["gemma4_unified"] = "gemma4"
        MODEL_REMAPPING["gemma4_unified_assistant"] = "gemma4"
        MODEL_REMAPPING["gemma4_assistant"] = "gemma4"
        
        # Download and cache model path first
        self.model_path = Path(_download(model_path_or_id))
        
        # Resolve config and apply dynamic patches
        self.config = self._load_and_patch_config()
        
        # Telemetry stats
        from tpm_mlx.speculation import SpeculationStats
        self.speculation_stats = SpeculationStats()

        if backend is not None and backend.lower() in ("llm", "vlm"):
            self.backend = backend.lower()
        else:
            self.backend = self._detect_backend(self.config)
        logger.info(f"Auto-detected inference backend: '{self.backend.upper()}' for {self.model_path_or_id}")
        
        if self.backend == "vlm":
            self._init_vlm_backend()
        else:
            self._init_llm_backend()

    def _detect_backend(self, config: Dict[str, Any]) -> str:
        """Auto-detects whether the model is a Multimodal VLM or Pure Text LLM."""
        vlm_model_types = {
            "gemma4", "gemma4_assistant", "gemma4_unified", "gemma4_unified_assistant",
            "qwen2_vl", "qwen2_5_vl", "llava", "llava_next", "pixtral",
            "paligemma", "idefics2", "florence2", "molmo", "internvl_chat", "phi3_v"
        }
        model_type = str(config.get("model_type", "")).lower()
        archs = [str(a).lower() for a in config.get("architectures", [])]
        
        if model_type in vlm_model_types:
            return "vlm"
        if any("vision" in a or "vlm" in a or "visual" in a or "llava" in a or "pixtral" in a for a in archs):
            return "vlm"
        if "vision_config" in config:
            return "vlm"
        return "llm"

    def _init_vlm_backend(self):
        """Initializes mlx-vlm model, processor, and optional MTP drafter."""
        import mlx_vlm.utils
        from mlx_vlm.speculative import load_drafter
        
        self.draft_model = None
        self.draft_kind = "none"
        self.mtp_head = None
        
        self.model, self.processor = mlx_vlm.utils.load(self.model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        
        if self.draft_model_path_or_id:
            try:
                loaded_drafter = load_drafter(self.draft_model_path_or_id, kind="mtp")
                if isinstance(loaded_drafter, tuple):
                    self.draft_model = loaded_drafter[0]
                    self.draft_kind = loaded_drafter[1]
                else:
                    self.draft_model = loaded_drafter
                    self.draft_kind = "mtp"
                logger.info(f"Successfully loaded VLM drafter '{self.draft_model_path_or_id}' (kind: {self.draft_kind})")
            except Exception as e:
                logger.warning(f"Failed to load VLM drafter via load_drafter: {e}. Falling back to standard generation.")
                self.draft_model = None
                self.draft_kind = "none"

    def _init_llm_backend(self):
        """Initializes mlx-lm model, tokenizer, and native MTP heads."""
        self.draft_model = None
        self.draft_kind = "none"
        self.processor = None
        self.mtp_head = None
        
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Load MTP head if enabled
        if self.enable_mtp:
            from tpm_mlx.mtp_weights import load_mtp_head_from_dir
            self.mtp_head = load_mtp_head_from_dir(self.model_path, self.model, self.config)

        # Load external draft model or standalone MTP head checkpoint if provided
        if self.draft_model_path_or_id:
            from tpm_mlx.mtp_weights import load_mtp_head_from_dir
            draft_path = Path(_download(self.draft_model_path_or_id))
            loaded_mtp = load_mtp_head_from_dir(draft_path, self.model, self.config)
            if loaded_mtp is not None:
                self.mtp_head = loaded_mtp
            else:
                self.draft_model = self._load_draft_model(self.draft_model_path_or_id)

    @property
    def has_mtp(self) -> bool:
        """Returns True if native MTP heads are loaded and active."""
        if getattr(self, "backend", "llm") == "vlm":
            return self.draft_model is not None and getattr(self, "draft_kind", "none") == "mtp"
        return self.mtp_head is not None

    @property
    def speculation_mode(self) -> str:
        """Returns active speculation strategy: 'mtp', 'draft', or 'none'."""
        if self.has_mtp:
            return "mtp"
        elif self.draft_model is not None:
            return "draft"
        return "none"

    @property
    def num_draft_tokens(self) -> int:
        """Resolves the number of tokens to speculate per step."""
        if self.user_num_draft_tokens is not None:
            return self.user_num_draft_tokens
        if getattr(self, "backend", "llm") == "vlm" and self.draft_model is not None:
            return 3
        if self.has_mtp and hasattr(self.mtp_head, "depth"):
            return self.mtp_head.depth
        return 2

    def _load_draft_model(self, draft_model_path_or_id: str):
        """Loads an external draft model for companion speculative decoding."""
        draft_path = Path(_download(draft_model_path_or_id))
        draft_config_path = draft_path / "config.json"
        draft_config = None
        if draft_config_path.exists():
            with open(draft_config_path, "r") as f:
                draft_config = json.load(f)
            if draft_config.get("model_type") == "gemma4_assistant":
                if "text_config" in draft_config and isinstance(draft_config["text_config"], dict):
                    if draft_config["text_config"].get("num_kv_shared_layers", 0) > 0:
                        draft_config["text_config"]["num_kv_shared_layers"] = 0
                        
        draft_model, _ = load_model(
            draft_path,
            lazy=False,
            strict=False,
            model_config=draft_config
        )
        return draft_model

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
            
        return config

    def _load_model_and_tokenizer(self):
        """
        Loads the tokenizer and model. Uses strict=False to bypass key mismatches.
        """
        model, _ = load_model(
            self.model_path, 
            lazy=False, 
            strict=False, 
            model_config=self.config
        )
        
        tokenizer = load_tokenizer(
            self.model_path, 
            tokenizer_config_extra=None, 
            eos_token_ids=self.config.get("eos_token_id", None)
        )
        
        return model, tokenizer

    def _stream_mtp_generate(
        self,
        prompt: Union[str, mx.array, List[int]],
        max_tokens: int = 4096,
        sampler: Optional[Callable] = None,
        prompt_cache: Optional[List[Any]] = None,
        num_draft_tokens: int = 2,
    ) -> Generator[GenerationResponse, None, None]:
        """Runs streaming MTP self-speculative generation."""
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from tpm_mlx.speculation import mtp_generate_step
        from mlx_lm.generate import generation_stream, wired_limit

        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)

        if not isinstance(prompt, mx.array):
            if isinstance(prompt, str):
                add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                    tokenizer.bos_token
                )
                prompt_arr = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            else:
                prompt_arr = prompt
            prompt_arr = mx.array(prompt_arr)
        else:
            prompt_arr = prompt

        detokenizer = tokenizer.detokenizer
        token_generator = mtp_generate_step(
            prompt=prompt_arr,
            model=self.model,
            mtp_head=self.mtp_head,
            num_draft_tokens=num_draft_tokens,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=prompt_cache,
            stats=self.speculation_stats,
        )

        with wired_limit(self.model, [generation_stream]):
            tic = time.perf_counter()
            token = 0
            logprobs = mx.array([])
            from_draft = False
            prompt_tps = 0.0
            n = 0

            for n, (token, logprobs, from_draft) in enumerate(token_generator):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = prompt_arr.size / max(prompt_time, 1e-6)
                    tic = time.perf_counter()
                if token in tokenizer.eos_token_ids:
                    break

                detokenizer.add_token(token)
                if (n + 1) == max_tokens:
                    break

                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    from_draft=from_draft,
                    prompt_tokens=prompt_arr.size,
                    prompt_tps=prompt_tps,
                    generation_tokens=n + 1,
                    generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )

            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt_arr.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            )

    def _stream_vlm_generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        images: Optional[List[Any]] = None,
    ) -> Generator[GenerationResponse, None, None]:
        """Runs streaming VLM generation using mlx-vlm."""
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        
        # Apply model chat template if string prompt
        formatted_prompt = prompt
        if hasattr(self, "processor") and hasattr(self.processor, "apply_chat_template"):
            try:
                formatted_prompt = apply_chat_template(self.processor, self.model.config, prompt)
            except Exception:
                formatted_prompt = prompt
            
        gen_kwargs = {
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            gen_kwargs["images"] = images
            
        if self.draft_model is not None:
            gen_kwargs["draft_model"] = self.draft_model
            gen_kwargs["draft_kind"] = getattr(self, "draft_kind", "mtp")
            gen_kwargs["draft_block_size"] = self.num_draft_tokens

        tic = time.perf_counter()
        n = 0
        prompt_tokens = len(self.tokenizer.encode(formatted_prompt)) if hasattr(self.tokenizer, "encode") else 1
        prompt_tps = 0.0

        for n, resp in enumerate(vlm_stream_generate(self.model, self.processor, **gen_kwargs)):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt_tokens / max(prompt_time, 1e-6)
                tic = time.perf_counter()
                
            elapsed = time.perf_counter() - tic
            gen_tps = (n + 1) / max(elapsed, 1e-6)
            
            yield GenerationResponse(
                text=resp.text,
                token=getattr(resp, "token", 0),
                logprobs=getattr(resp, "logprobs", mx.array([])),
                from_draft=getattr(resp, "from_draft", False),
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=gen_tps,
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

    def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 4096, 
        temperature: float = 0.0,
        show_reasoning: bool = False,
        images: Optional[List[Any]] = None,
    ) -> Generator[GenerationResponse, None, None]:
        """
        A streaming generator wrapping speculative / standard generation with PreAllocatedKVCache
        and reasoning filtering.
        """
        # If VLM backend is active, route through mlx-vlm stream generator
        if self.backend == "vlm":
            raw_stream = self._stream_vlm_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                images=images,
            )
            if show_reasoning:
                yield from self._normalize_reasoning(raw_stream)
            else:
                yield from self._filter_reasoning(raw_stream)
            return

        # Construct pre-allocated prompt cache for LLM backend
        from mlx_lm.models import cache as mlx_cache
        prompt_cache = mlx_cache.make_prompt_cache(self.model, max_kv_size=self.max_kv_size)
        for i, c in enumerate(prompt_cache):
            if type(c) is KVCache:
                prompt_cache[i] = PreAllocatedKVCache(max_size=self.max_kv_size)
        
        # Build generation keyword arguments
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)
        
        # Route to appropriate backend
        if self.speculation_mode == "mtp":
            raw_stream = self._stream_mtp_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
                num_draft_tokens=self.num_draft_tokens,
            )
        elif self.speculation_mode == "draft":
            gen_kwargs = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "prompt_cache": prompt_cache,
                "draft_model": self.draft_model,
                "num_draft_tokens": self.num_draft_tokens,
            }
            raw_stream = stream_generate(self.model, self.tokenizer, prompt, **gen_kwargs)
        else:
            gen_kwargs = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "prompt_cache": prompt_cache,
            }
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
