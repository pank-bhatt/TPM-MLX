# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import os
import copy
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from tpm_mlx.mtp import MTPHead, MTPLayer
from tpm_mlx.utils import get_logger

logger = get_logger("mtp_weights")


def load_raw_safetensors(model_dir: Union[str, Path]) -> Dict[str, mx.array]:
    """
    Loads all raw tensor weights from .safetensors files in the model directory
    without running mlx-lm's sanitize() filtering.
    """
    model_path = Path(model_dir)
    safetensor_files = list(model_path.glob("*.safetensors"))
    
    if not safetensor_files:
        return {}
        
    weights: Dict[str, mx.array] = {}
    for sf in safetensor_files:
        try:
            loaded = mx.load(str(sf))
            weights.update(loaded)
        except Exception as e:
            logger.warning(f"Could not load tensors from {sf}: {e}")
            
    return weights


def detect_mtp_family(weights: Dict[str, Any]) -> Optional[str]:
    """
    Detects which model family's MTP head structure is present in the raw weights.
    """
    if any(k.startswith("mtp.") or k.startswith("model.mtp.") for k in weights):
        return "qwen3"
    elif any("pre_fc_norm_embedding" in k or "pre_fc_norm_hidden" in k for k in weights):
        return "qwen3_standalone"
    elif any(k.startswith("model.mtp_layers.") for k in weights):
        return "mimo"
    elif any("mtp_block." in k or "mtp_linear_proj." in k for k in weights):
        return "ernie_mimo"
    elif any(k.startswith("model.layers.61.") for k in weights):
        return "deepseek_v3"
    return None


def extract_and_remap_mtp_weights(
    weights: Dict[str, mx.array], 
    family: str,
    hidden_size: int,
) -> Tuple[Dict[str, mx.array], int]:
    """
    Remaps raw checkpoint keys into canonical MTPHead parameter keys:
        layers.{i}.enorm.weight
        layers.{i}.hnorm.weight
        layers.{i}.proj.weight
        layers.{i}.block.*
        layers.{i}.norm.weight
    """
    remapped: Dict[str, mx.array] = {}
    detected_depths = set()

    if family in ("qwen3", "qwen3_standalone"):
        for k, v in weights.items():
            if family == "qwen3_standalone":
                # Keys are directly named: pre_fc_norm_embedding.weight, pre_fc_norm_hidden.weight, fc.weight, layers.0.*, norm.weight
                if k == "pre_fc_norm_embedding.weight":
                    remapped["layers.0.enorm.weight"] = v
                    detected_depths.add(0)
                elif k == "pre_fc_norm_hidden.weight":
                    remapped["layers.0.hnorm.weight"] = v
                    detected_depths.add(0)
                elif k.startswith("fc."):
                    sub = k.replace("fc.", "")
                    remapped[f"layers.0.proj.{sub}"] = v
                    detected_depths.add(0)
                elif k.startswith("norm."):
                    sub = k.replace("norm.", "")
                    remapped[f"layers.0.norm.{sub}"] = v
                    detected_depths.add(0)
                elif k.startswith("layers."):
                    parts = k.split(".")
                    layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    detected_depths.add(layer_idx)
                    sub_key = ".".join(parts[2:])
                    remapped[f"layers.{layer_idx}.block.{sub_key}"] = v
                continue

            if not (k.startswith("mtp.") or k.startswith("model.mtp.")):
                continue
                
            clean_k = k.replace("model.mtp.", "mtp.")
            
            # Format: mtp.layers.{i}.* or mtp.{layer_idx}.* or mtp.enorm.weight
            parts = clean_k.split(".")
            # Determine layer index
            layer_idx = 0
            if len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit():
                layer_idx = int(parts[2])
                detected_depths.add(layer_idx)
                sub_key = ".".join(parts[3:])
            elif len(parts) > 1 and parts[1].isdigit():
                layer_idx = int(parts[1])
                detected_depths.add(layer_idx)
                sub_key = ".".join(parts[2:])
            else:
                sub_key = ".".join(parts[1:])
                detected_depths.add(0)

            # Map sub_keys
            if sub_key in ("enorm.weight", "input_enorm.weight", "embed_norm.weight", "pre_fc_norm_embedding.weight"):
                remapped[f"layers.{layer_idx}.enorm.weight"] = v
            elif sub_key in ("hnorm.weight", "input_hnorm.weight", "hidden_norm.weight", "pre_fc_norm_hidden.weight"):
                remapped[f"layers.{layer_idx}.hnorm.weight"] = v
            elif sub_key in ("eh_proj.weight", "input_proj.weight", "fc.weight", "proj.weight"):
                remapped[f"layers.{layer_idx}.proj.weight"] = v
            elif sub_key in ("norm.weight", "final_norm.weight"):
                remapped[f"layers.{layer_idx}.norm.weight"] = v
            else:
                # Transformer block sub-layers
                remapped[f"layers.{layer_idx}.block.{sub_key}"] = v

    elif family in ("mimo", "ernie_mimo"):
        for k, v in weights.items():
            if not k.startswith("model.mtp_layers."):
                continue
            # Format: model.mtp_layers.{i}.*
            parts = k.replace("model.mtp_layers.", "").split(".")
            if parts[0].isdigit():
                layer_idx = int(parts[0])
                detected_depths.add(layer_idx)
                sub_key = ".".join(parts[1:])
                
                if sub_key == "enorm.weight":
                    remapped[f"layers.{layer_idx}.enorm.weight"] = v
                elif sub_key == "hnorm.weight":
                    remapped[f"layers.{layer_idx}.hnorm.weight"] = v
                elif sub_key in ("input_proj.weight", "proj.weight"):
                    remapped[f"layers.{layer_idx}.proj.weight"] = v
                elif sub_key == "norm.weight":
                    remapped[f"layers.{layer_idx}.norm.weight"] = v
                else:
                    remapped[f"layers.{layer_idx}.block.{sub_key}"] = v

    elif family == "deepseek_v3":
        # Deepseek-V3 MTP module is typically layer 61
        for k, v in weights.items():
            if not k.startswith("model.layers.61."):
                continue
            detected_depths.add(0)
            sub_key = k.replace("model.layers.61.", "")
            if "enorm" in sub_key:
                remapped["layers.0.enorm.weight"] = v
            elif "hnorm" in sub_key:
                remapped["layers.0.hnorm.weight"] = v
            elif "eh_proj" in sub_key or "proj" in sub_key:
                remapped["layers.0.proj.weight"] = v
            elif "norm" in sub_key:
                remapped["layers.0.norm.weight"] = v
            else:
                remapped[f"layers.0.block.{sub_key}"] = v

    depth = max(detected_depths) + 1 if detected_depths else 0
    return remapped, depth


def load_mtp_head_from_dir(
    model_dir: Union[str, Path],
    base_model: nn.Module,
    config: Dict[str, Any],
) -> Optional[MTPHead]:
    """
    Scans the checkpoint directory for MTP weights, initializes the MTPHead,
    and loads weights into it.
    
    Returns:
        MTPHead instance if MTP weights are present and valid, else None.
    """
    raw_weights = load_raw_safetensors(model_dir)
    if not raw_weights:
        return None
        
    family = detect_mtp_family(raw_weights)
    if not family:
        logger.debug("No MTP weights detected in checkpoint.")
        return None
        
    logger.info(f"Detected MTP weights with architecture family '{family}'")
    
    # Resolve dimensions
    hidden_size = config.get("hidden_size")
    if hidden_size is None and "text_config" in config:
        hidden_size = config["text_config"].get("hidden_size")
    if hidden_size is None and hasattr(base_model, "args"):
        hidden_size = getattr(base_model.args, "hidden_size", None)
    if hidden_size is None:
        hidden_size = 4096  # fallback default
        
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)
    
    # Extract remapped weights
    mtp_weights, depth = extract_and_remap_mtp_weights(
        raw_weights, family=family, hidden_size=hidden_size
    )
    
    if depth == 0 or not mtp_weights:
        logger.warning("Found MTP keys but failed to extract layers.")
        return None
        
    logger.info(f"Initializing MTPHead with depth={depth}, hidden_size={hidden_size}")
    
    # Extract embed_tokens and lm_head from base_model
    embed_tokens = None
    if hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
        embed_tokens = base_model.model.embed_tokens
    elif hasattr(base_model, "embed_tokens"):
        embed_tokens = base_model.embed_tokens
    elif hasattr(base_model, "language_model") and hasattr(base_model.language_model.model, "embed_tokens"):
        embed_tokens = base_model.language_model.model.embed_tokens
        
    lm_head = None
    if hasattr(base_model, "lm_head"):
        lm_head = base_model.lm_head
    elif hasattr(base_model, "language_model") and hasattr(base_model.language_model, "lm_head"):
        lm_head = base_model.language_model.lm_head
    elif embed_tokens is not None and hasattr(embed_tokens, "as_linear"):
        lm_head = embed_tokens.as_linear
        
    if embed_tokens is None or lm_head is None:
        logger.warning("Could not resolve shared embed_tokens or lm_head from base model.")
        return None
        
    # Instantiate MTP layers (clone a decoder layer matching attention type as template)
    layers = []
    has_quant_proj = any("proj.scales" in k for k in mtp_weights)
    needs_self_attn = any("self_attn" in k for k in mtp_weights)
    needs_linear_attn = any("linear_attn" in k or "conv1d" in k for k in mtp_weights)
    
    all_layers = []
    if hasattr(base_model, "layers") and len(base_model.layers) > 0:
        all_layers = base_model.layers
    elif hasattr(base_model, "model") and hasattr(base_model.model, "layers") and len(base_model.model.layers) > 0:
        all_layers = base_model.model.layers
    elif hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers") and len(base_model.language_model.layers) > 0:
        all_layers = base_model.language_model.layers
    elif hasattr(base_model, "language_model") and hasattr(base_model.language_model, "model") and hasattr(base_model.language_model.model, "layers"):
        all_layers = base_model.language_model.model.layers

    template_layer = None
    if needs_self_attn:
        for l in all_layers:
            if hasattr(l, "self_attn"):
                template_layer = l
                break
    elif needs_linear_attn:
        for l in all_layers:
            if hasattr(l, "linear_attn"):
                template_layer = l
                break
                
    if template_layer is None and len(all_layers) > 0:
        template_layer = all_layers[-1]

    for i in range(depth):
        block = copy.deepcopy(template_layer) if template_layer is not None else None
            
        layer = MTPLayer(
            hidden_size=hidden_size,
            rms_norm_eps=rms_norm_eps,
            block=block,
        )
        
        if has_quant_proj:
            layer.proj = nn.QuantizedLinear.from_linear(layer.proj, group_size=64, bits=4)
            
        layers.append(layer)
        
    mtp_head = MTPHead(
        hidden_size=hidden_size,
        depth=depth,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        rms_norm_eps=rms_norm_eps,
        layers=layers,
    )
    
    # Load extracted weights into the head using tree_unflatten
    try:
        from mlx.utils import tree_unflatten
        unflattened = tree_unflatten(list(mtp_weights.items()))
        mtp_head.update(unflattened)
        logger.info(f"Successfully loaded {len(mtp_weights)} MTP parameter tensors into MTPHead.")
        return mtp_head
    except Exception as e:
        logger.warning(f"Error loading MTP weights into MTPHead: {e}")
        return None
