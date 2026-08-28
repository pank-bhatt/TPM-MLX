# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache

from tpm_mlx.mtp import MTPHead
from tpm_mlx.utils import get_logger

logger = get_logger("speculation")


@dataclass
class SpeculationStats:
    """Telemetry data collected during speculative generation."""
    draft_tokens_total: int = 0
    accepted_tokens_total: int = 0
    speculation_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_total == 0:
            return 0.0
        return self.accepted_tokens_total / self.draft_tokens_total

    @property
    def avg_tokens_per_step(self) -> float:
        if self.speculation_steps == 0:
            return 1.0
        # Total generated tokens is accepted draft tokens + 1 target token per step
        total_tokens = self.accepted_tokens_total + self.speculation_steps
        return total_tokens / self.speculation_steps


def mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    mtp_head: MTPHead,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    stats: Optional[SpeculationStats] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """
    Self-speculative token generator using built-in MTP heads.

    Yields:
        Tuple[int, mx.array, bool]: (token_id, logprobs, from_draft)
    """
    y = prompt.astype(mx.uint32)
    prev_tokens = None

    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
    else:
        model_cache = prompt_cache[: len(model.layers)]

    # Support both trimmable attention caches and snapshot-able recurrent caches (like ArraysCache in Qwen 3.5/3.8)
    has_unsupported_cache = any(
        not c.is_trimmable() and not hasattr(c, "state") for c in model_cache
    )
    if has_unsupported_cache:
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable() and not hasattr(c, "state")}
        raise ValueError(f"MTP speculative decoding requires trimmable or stateful prompt caches (got {types}).")

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = sampler(logprobs)
        return sampled, logprobs

    def _model_forward(input_tokens, target_cache):
        """Runs forward through trunk to get both hidden states and output logits."""
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            h = model.language_model.model(input_tokens, cache=target_cache)
            if hasattr(model.language_model, "lm_head"):
                logits = model.language_model.lm_head(h)
            elif hasattr(model.language_model.model, "embed_tokens") and hasattr(model.language_model.model.embed_tokens, "as_linear"):
                logits = model.language_model.model.embed_tokens.as_linear(h)
            else:
                logits = model.language_model(input_tokens, cache=target_cache)
        elif hasattr(model, "model"):
            h = model.model(input_tokens, cache=target_cache)
            if hasattr(model, "lm_head"):
                logits = model.lm_head(h)
            elif hasattr(model.model, "embed_tokens") and hasattr(model.model.embed_tokens, "as_linear"):
                logits = model.model.embed_tokens.as_linear(h)
            else:
                logits = model(input_tokens, cache=target_cache)
        else:
            logits = model(input_tokens, cache=target_cache)
            h = logits
        return logits, h

    def _step(target_model, target_cache, input_tokens, n_predict=1):
        with mx.stream(generation_stream):
            logits, h = _model_forward(input_tokens[None], target_cache)
            logits = logits[:, -n_predict:, :]
            last_h = h[:, -1:, :]
            quantize_cache_fn(target_cache)

            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                curr_input = input_tokens
                if n_predict > 1:
                    curr_input = curr_input[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, curr_input])
                        if prev_tokens is not None
                        else curr_input
                    )
                    sampled_tok, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(sampled_tok)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0), last_h
            else:
                sampled_tok, logprobs = _process_and_sample(None, logits.squeeze(0))
                return sampled_tok, logprobs, last_h

    def _prefill(target_model, target_cache, input_tokens):
        last_h = None
        while input_tokens.size > 1:
            n_to_process = min(prefill_step_size, input_tokens.size - 1)
            _, h = _model_forward(input_tokens[:n_to_process][None], target_cache)
            last_h = h[:, -1:, :]
            quantize_cache_fn(target_cache)
            states = [c.state for c in target_cache if hasattr(c, "empty") and not c.empty()]
            if states:
                mx.eval(states)
            input_tokens = input_tokens[n_to_process:]
            mx.clear_cache()
        return input_tokens, last_h

    def _snapshot_recurrent():
        return {
            i: [s for s in c.state] if isinstance(c.state, (list, tuple)) else c.state
            for i, c in enumerate(model_cache)
            if hasattr(c, "state") and not c.is_trimmable()
        }

    def _rewind_cache(num_draft, num_accept, recurrent_snapshot=None, accepted_prefix=None):
        if num_draft > num_accept:
            cache.trim_prompt_cache(model_cache, num_draft - num_accept)
            if recurrent_snapshot:
                for i, state in recurrent_snapshot.items():
                    model_cache[i].state = state
                if accepted_prefix is not None and accepted_prefix.size > 0:
                    model(accepted_prefix[None], cache=model_cache)

    def _mtp_draft(curr_token, last_hidden, num_draft):
        """Uses MTPHead to speculate candidate future tokens."""
        if num_draft <= 0 or last_hidden is None:
            return mx.array([], mx.uint32)
            
        with mx.stream(generation_stream):
            drafted_tokens, _, _ = mtp_head.draft_sequence(
                hidden_state=last_hidden,
                initial_token=curr_token,
                num_draft_tokens=num_draft,
                sampler=sampler,
            )
            if not drafted_tokens:
                return mx.array([], mx.uint32)
            return mx.concatenate(drafted_tokens)

    # 1. Prefill prompt
    with mx.stream(generation_stream):
        y, last_h = _prefill(model, model_cache, y)
        if last_h is None:
            _, last_h = _model_forward(y[None], model_cache)

    ntoks = 0
    num_draft = 0
    n = 0
    
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            recurrent_snapshot = _snapshot_recurrent()
            draft_tokens = _mtp_draft(y, last_h, num_draft)
            
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
                
            y_verify = mx.concatenate([y, draft_tokens]) if draft_tokens.size > 0 else y
            tokens, logprobs, last_h = _step(model, model_cache, y_verify, draft_tokens.size + 1)
            
            mx.eval(tokens, draft_tokens)
            draft_list = draft_tokens.tolist()
            tokens_list = tokens.tolist() if hasattr(tokens, "tolist") else [tokens.item()]
            
            if stats is not None:
                stats.speculation_steps += 1
                stats.draft_tokens_total += len(draft_list)
                
            n = 0
            while n < len(draft_list):
                tn, dtn, lpn = tokens_list[n], draft_list[n], logprobs[n]
                if tn != dtn:
                    break
                n += 1
                ntoks += 1
                if stats is not None:
                    stats.accepted_tokens_total += 1
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break
                    
            if ntoks < max_tokens and n < len(tokens_list):
                ntoks += 1
                yield tokens_list[n], logprobs[n], False

            if ntoks >= max_tokens:
                break

            last_valid_token = tokens_list[n] if n < len(tokens_list) else tokens_list[-1]
            accepted_prefix = y_verify[1 : n + 1] if n > 0 else None
            _rewind_cache(len(draft_list), n, recurrent_snapshot, accepted_prefix)
            y = mx.array([last_valid_token], mx.uint32)

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(len(draft_list) - n, 1)]
            
    finally:
        _rewind_cache(num_draft, n)


def draft_model_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    stats: Optional[SpeculationStats] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """
    Speculative generator using an external companion draft/assistant model (e.g. Gemma 4 assistant).
    """
    from mlx_lm.generate import speculative_generate_step as mlx_spec_step

    raw_gen = mlx_spec_step(
        prompt=prompt,
        model=model,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )

    for token, logprobs, from_draft in raw_gen:
        if stats is not None:
            if from_draft:
                stats.accepted_tokens_total += 1
                stats.draft_tokens_total += 1
            else:
                stats.speculation_steps += 1
        yield int(token), logprobs, from_draft
