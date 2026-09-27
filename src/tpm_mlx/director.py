# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
AI Visual Director prompt distillation engine.
Transforms conversational context and concise user descriptions into dense,
cinematic visual prompts tailored for Apple Silicon diffusion models.
"""

import asyncio
import logging
from typing import Optional

import mlx.core as mx

from tpm_mlx import state

logger = logging.getLogger("tpm-mlx.director")


async def distill_visual_prompt(
    prompt: str,
    context: Optional[str] = None,
    is_edit: bool = False,
) -> str:
    """
    Synthesizes an enriched, photorealistic visual prompt from conversational context
    using the active LLM engine and its native chat template.
    """
    if state.engine is None:
        return prompt

    loop = asyncio.get_running_loop()

    def distill() -> str:
        if is_edit:
            sys_msg = (
                "You are an expert visual prompt director for image editing. "
                "Analyze the edit request and context, then synthesize a concise, precise image modification prompt "
                "focusing on the exact changes to be made (clothing, colors, lighting, objects) while retaining unchanged elements (max 60 words). "
                "Output ONLY the prompt text."
            )
            usr_msg = f"Context:\n{context or ''}\n\nEdit Request: {prompt}"
        else:
            sys_msg = (
                "You are an expert visual prompt director. "
                "Analyze the context and request, then synthesize a single dense, photorealistic "
                "image generation prompt focusing on subjects, lighting, colors, mood, and composition (max 80 words). "
                "Output ONLY the prompt text."
            )
            usr_msg = f"Context:\n{context or ''}\n\nRequest: {prompt}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": usr_msg},
        ]

        distill_prompt_text = ""
        engine = state.engine
        if engine and hasattr(engine, "tokenizer") and hasattr(engine.tokenizer, "apply_chat_template"):
            try:
                distill_prompt_text = engine.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                distill_prompt_text = ""

        if not distill_prompt_text:
            distill_prompt_text = f"System: {sys_msg}\nUser: {prompt}\nAssistant: "

        try:
            responses = list(engine.generate_stream(distill_prompt_text, max_tokens=150, temperature=0.3, show_reasoning=False))
            text = "".join(r.text for r in responses).strip()
        except Exception as e:
            logger.warning(f"Error during prompt distillation generation: {e}")
            return prompt

        # Strip thinking / channel / turn tags
        for tag in ("</think>", "<channel|>", "<|channel|>", "<|channel>", "<end_of_turn>", "<|im_end|>", "<|end_of_text|>"):
            if tag in text:
                text = text.split(tag)[-1].strip()

        # Clean up common prefixes like "Prompt: ..."
        if ":" in text and len(text.split(":", 1)[0]) < 60:
            prefix = text.split(":", 1)[0].lower()
            if any(k in prefix for k in ("prompt", "output", "description", "image", "visual prompt")):
                text = text.split(":", 1)[1].strip()

        # Remove leftover special token markers
        for garbage in ("<|channel>", "<|channel|>", "<channel|>", "<|im_start|>", "<start_of_turn>"):
            text = text.replace(garbage, "").strip()

        if not text or len(text) < 10:
            return prompt

        mx.synchronize()
        return text

    try:
        revised = await loop.run_in_executor(state.mlx_executor, distill)
        return revised if revised and len(revised) >= 10 else prompt
    except Exception as ex:
        logger.warning(f"Prompt distillation failed: {ex}")
        return prompt


# Backwards compatibility alias
_distill_visual_prompt = distill_visual_prompt
