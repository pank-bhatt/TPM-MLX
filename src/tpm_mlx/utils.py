# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from huggingface_hub import scan_cache_dir

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tpm-mlx")


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the configured prefix."""
    return logging.getLogger(f"tpm-mlx.{name}")


def get_cached_models() -> List[Dict[str, Any]]:
    """
    Scans Hugging Face cache and returns a list of cached models.
    """
    models = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type == "model":
                models.append({
                    "repo_id": repo.repo_id,
                    "size_on_disk": repo.size_on_disk,
                    "last_modified": repo.last_modified,
                    "local_path": str(repo.repo_path)
                })
    except Exception as e:
        logger.warning(f"Could not scan Hugging Face cache directory: {e}")
        
    return models


def apply_chat_template_fallback(
    messages: List[Dict[str, str]], 
    tokenizer: Any
) -> str:
    """
    Tries to apply chat templates to the messages list.
    Falls back to a standard format if tokenizer.chat_template is not available.
    """
    try:
        # Check if tokenizer has apply_chat_template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logger.debug(f"apply_chat_template failed: {e}. Falling back to default formatting.")
        
    # Default chat template fallback formatting
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user").strip().lower()
        content = msg.get("content", "").strip()
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
            
    prompt += "<|assistant|>\n"
    return prompt
