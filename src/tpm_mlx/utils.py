# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from huggingface_hub import scan_cache_dir


class SafeStream:
    """Wraps a stream to silently ignore BrokenPipeError and related OSErrors (e.g. when pipes close on background tasks)."""
    def __init__(self, target):
        self._target = target

    def write(self, s):
        try:
            return self._target.write(s)
        except (BrokenPipeError, OSError):
            return len(s) if hasattr(s, "__len__") else 0

    def flush(self):
        try:
            return self._target.flush()
        except (BrokenPipeError, OSError):
            pass

    def isatty(self):
        try:
            return self._target.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self._target.fileno()
        except Exception:
            raise OSError("SafeStream has no fileno")

    def __getattr__(self, attr):
        return getattr(self._target, attr)


def install_safe_streams():
    """Protects sys.stdout and sys.stderr against BrokenPipeError."""
    if not isinstance(sys.stdout, SafeStream):
        sys.stdout = SafeStream(sys.stdout)
    if not isinstance(sys.stderr, SafeStream):
        sys.stderr = SafeStream(sys.stderr)


# Automatically protect streams
install_safe_streams()

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


import json

def get_cached_models() -> List[Dict[str, Any]]:
    """
    Scans Hugging Face cache and returns only valid, locally downloaded models
    compatible with TPM-MLX (generative LLMs, MTP assistants, and image generation models).
    Filters out non-MLX, non-generative, audio (whisper), masked-LM, decoder subcomponents, or empty repos.
    """
    models = []
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return models

    skip_archs = {
        "whisper",
        "xlmrobertaformaskedlm",
        "mbartforconditionalgeneration",
        "bertformaskedlm",
        "robertaformaskedlm",
    }

    try:
        for d in sorted(cache_dir.iterdir()):
            if not d.name.startswith("models--"):
                continue
            repo_id = d.name.removeprefix("models--").replace("--", "/")
            snaps_dir = d / "snapshots"
            if not snaps_dir.exists():
                continue
            snaps = [p for p in snaps_dir.iterdir() if p.is_dir()]
            if not snaps:
                continue
            latest = max(snaps, key=lambda p: p.stat().st_mtime)
            files = list(latest.iterdir())
            file_names = {f.name for f in files}

            # Must have multiple files, not empty/incomplete
            if not files or len(files) <= 1:
                continue

            # Skip sub-components that aren't full models
            if "decoder" in repo_id.lower() or ("encoder" in repo_id.lower() and not any(k in file_names for k in ("text_encoder", "transformer", "config.json"))):
                continue

            # Check config.json for non-generative architectures
            config_path = latest / "config.json"
            arch = ""
            model_type = ""
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    archs = cfg.get("architectures", [])
                    arch = archs[0] if isinstance(archs, list) and archs else ""
                    model_type = cfg.get("model_type", "")
                except Exception:
                    pass

            if model_type.lower() in skip_archs or arch.lower() in skip_archs:
                continue

            # Check for MLX weights (.safetensors or .npz) in root or subdirectories
            has_weights = any(f.suffix in (".safetensors", ".npz") for f in files) or any(
                sub.is_dir() and any(sf.suffix in (".safetensors", ".npz") for sf in sub.iterdir())
                for sub in files if sub.is_dir()
            )
            if not has_weights:
                continue

            is_draft = (
                "assistant" in repo_id.lower()
                or "-mtp" in repo_id.lower()
                or "_mtp" in repo_id.lower()
                or "mtp" in arch.lower()
                or "assistant" in arch.lower()
            )
            is_image = (
                any(k in repo_id.lower() for k in ("flux", "diffusion", "klein", "sdxl", "stable-diffusion", "z_image", "z-image", "zimage"))
                or ("bonsai" in repo_id.lower() and "image" in repo_id.lower())
            )

            size_bytes = 0
            try:
                for f in files:
                    if f.is_file():
                        size_bytes += f.stat().st_size
                    elif f.is_dir():
                        for sf in f.iterdir():
                            if sf.is_file():
                                size_bytes += sf.stat().st_size
            except Exception:
                size_bytes = 0

            models.append({
                "repo_id": repo_id,
                "size_on_disk": size_bytes,
                "last_modified": latest.stat().st_mtime,
                "local_path": str(latest),
                "is_draft": is_draft,
                "is_image": is_image,
                "arch": arch or model_type,
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
