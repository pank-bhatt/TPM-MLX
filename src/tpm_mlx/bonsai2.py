"""
TPM-MLX: Native Bonsai 2 / Prism Hadamard Architecture Support.

Provides native MLX implementation of Hadamard-transformed ternary layers (fwht)
and dynamic model assembly for prism_hadamard_qwen35 (Ternary Bonsai 2 27B).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("tpm-mlx.bonsai2")


def fwht(x: mx.array, block: int, signs: Optional[mx.array] = None, inverse: bool = False) -> mx.array:
    """
    Fast Walsh-Hadamard Transform (FWHT) for rotated weight basis.

    Args:
        x: Input tensor of activations.
        block: Hadamard block dimension (e.g. 1024).
        signs: Vector of signs (+1/-1) applied before/after transform.
        inverse: If True, applies signs after transform (inverse FWHT).
    """
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block != 0:
        raise ValueError(f"Hadamard block size ({block}) does not divide activation width ({shape[-1]})")

    x = x.astype(mx.float32)
    if not inverse and signs is not None:
        x = x * signs

    # Reshape into blocks of size 'block' and apply normalized Hadamard transform
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1.0 / math.sqrt(block)).reshape(shape)

    if inverse and signs is not None:
        x = x * signs

    return x.astype(dtype)


class Packed(nn.Module):
    """
    Packed Ternary affine layer with group-wise scale/bias and optional Hadamard rotation.
    Decodes ternary weights {-s, 0, +s} via 2-bit container directly in Apple Silicon Metal.
    """
    def __init__(
        self,
        arrays: Tuple[Any, Any, Any],
        block: int = 0,
        signs: Optional[Any] = None,
        embedding: bool = False,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.weight = mx.array(arrays[0])
        self.scales = mx.array(arrays[1])
        self.biases = mx.array(arrays[2])
        self.block = block
        self.signs = mx.array(signs) if signs is not None else None
        self.embedding = embedding
        self.dtype = dtype

    def __call__(self, x: mx.array) -> mx.array:
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = mx.dequantize(
                self.weight[indices],
                self.scales[indices],
                self.biases[indices],
                group_size=128,
                bits=2,
            ).reshape(*shape, -1).astype(self.dtype)
            return fwht(out, self.block, self.signs, inverse=True) if self.block else out

        if self.block:
            x = fwht(x, self.block, self.signs)

        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )


def build_bonsai2_processor(directory: Path):
    """
    Constructs the Qwen3VL multimodal processor without importing torch torchvision.
    Uses PIL-backed image processing and fast huggingface tokenization.
    """
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from mlx_vlm.models.qwen3_5 import Qwen3VLProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    chat_template_path = directory / "chat_template.jinja"
    chat_template = chat_template_path.read_text() if chat_template_path.exists() else None

    image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(directory))
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=chat_template,
    )
    processor.detokenizer = load_tokenizer(directory, return_tokenizer=False)(tokenizer)
    eos = getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None)
    processor.stopping_criteria = StoppingCriteria(eos, tokenizer)
    processor.tokenizer.stopping_criteria = processor.stopping_criteria
    return processor


def chat_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """mlx-vlm's prompt helper keys off `model_type`; provide the base type so tokens land."""
    return {**config, "model_type": config.get("base_model_type", "qwen3_5")}



def load_bonsai2_model(
    model_path_or_id: Union[str, Path],
    load_processor: bool = True
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads a Ternary Bonsai 2 model (prism_hadamard_qwen35) into Apple Silicon Unified Memory.
    Wires Packed Hadamard linear layers and attaches the FP16 vision tower.

    Returns:
        (model, processor, config)
    """
    from mlx_lm.utils import _download
    directory = Path(_download(str(model_path_or_id)))
    
    config_file = directory / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {directory}")

    config = json.loads(config_file.read_text())
    model_type = config.get("model_type", "")
    if model_type != "prism_hadamard_qwen35":
        raise ValueError(f"Unsupported model_type '{model_type}'. Expected 'prism_hadamard_qwen35'.")

    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    logger.info(f"Instantiating Bonsai 2 base architecture (Qwen3.8-27B)...")
    model = Model(ModelConfig.from_dict(config))

    weights_file = directory / "model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {directory}")

    logger.info(f"Loading weights from {weights_file.name} (8.6 GB)...")
    weights = mx.load(str(weights_file))

    # Install Packed Hadamard layers into language_model
    lm = model.language_model
    seen = set()
    modules = config.get("modules", [])
    logger.info(f"Wiring {len(modules)} Hadamard-rotated ternary layers...")

    for record in modules:
        path = record["path"]
        if path in seen:
            raise ValueError(f"Duplicate packed module path: {path}")
        seen.add(path)

        parts = path.split(".")
        parent = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

        key = "language_model." + path
        arrays = (
            weights[key + ".weight"],
            weights[key + ".scales"],
            weights[key + ".biases"],
        )
        block = record.get("block", 0)
        signs = weights.get(key + ".signs")
        embedding = record.get("embedding", False)

        packed_layer = Packed(arrays, block=block, signs=signs, embedding=embedding, dtype=mx.float16)
        setattr(parent, parts[-1], packed_layer)

    # Load remaining weights (normalization, linear attention states, vision tower)
    logger.info("Loading remaining model weights (vision tower & normalizations)...")
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())

    processor = build_bonsai2_processor(directory) if load_processor else None
    logger.info("Bonsai 2 model loaded successfully!")
    return model, processor, config
