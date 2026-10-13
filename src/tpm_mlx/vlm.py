# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
VLM and multimodal helpers for TPM-MLX.
Provides parsing of OpenAI-compatible vision payloads (data URIs, URLs, and image objects).
"""

import io
import base64
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("tpm-mlx.vlm")


def extract_text_and_images(messages: List[Any]) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Extracts text messages and PIL images from OpenAI multimodal messages format.
    Handles plain string content, lists of text/image_url dictionaries,
    base64 data URIs, and remote HTTP URLs.
    """
    formatted_messages = []
    images = []

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")
            tool_call_id = m.get("tool_call_id")
            name = m.get("name")
        else:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            tool_call_id = getattr(m, "tool_call_id", None)
            name = getattr(m, "name", None)

        msg_dict: Dict[str, Any] = {"role": role}
        if tool_calls is not None:
            msg_dict["tool_calls"] = [
                tc if isinstance(tc, dict) else (tc.model_dump() if hasattr(tc, "model_dump") else tc.__dict__)
                for tc in tool_calls
            ]
        if tool_call_id is not None:
            msg_dict["tool_call_id"] = tool_call_id
        if name is not None:
            msg_dict["name"] = name

        if content is None:
            msg_dict["content"] = ""
            formatted_messages.append(msg_dict)
        elif isinstance(content, str):
            msg_dict["content"] = content
            formatted_messages.append(msg_dict)
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_info = part.get("image_url", {})
                        img_url = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                        if img_url:
                            try:
                                from PIL import Image
                                from pathlib import Path
                                from tpm_mlx import state

                                # Check local disk paths first to avoid loopback HTTP request deadlocks
                                resolved_local = None
                                if "/images/uploads/" in img_url:
                                    fname = img_url.split("/images/uploads/")[-1].split("?")[0]
                                    cand = state.images_dir / "uploads" / fname
                                    if cand.exists():
                                        resolved_local = cand
                                elif "/images/" in img_url:
                                    fname = img_url.split("/images/")[-1].split("?")[0]
                                    cand = state.images_dir / fname
                                    if cand.exists():
                                        resolved_local = cand
                                elif "/static/" in img_url:
                                    fname = img_url.split("/static/")[-1].split("?")[0]
                                    cand = state.static_dir / fname
                                    if cand.exists():
                                        resolved_local = cand
                                else:
                                    cand_direct = Path(img_url)
                                    if cand_direct.exists():
                                        resolved_local = cand_direct

                                if resolved_local and resolved_local.exists():
                                    images.append(Image.open(resolved_local))
                                elif img_url.startswith("data:image"):
                                    header, base64_data = img_url.split(",", 1)
                                    img_bytes = base64.b64decode(base64_data)
                                    images.append(Image.open(io.BytesIO(img_bytes)))
                                elif img_url.startswith("http://") or img_url.startswith("https://"):
                                    import requests
                                    resp = requests.get(img_url, timeout=10)
                                    images.append(Image.open(io.BytesIO(resp.content)))
                                else:
                                    logger.warning(f"Could not resolve local image path from: '{img_url}'")
                            except Exception as ex:
                                logger.warning(f"Failed to load image from {img_url[:30]}...: {ex}")
            msg_dict["content"] = " ".join(text_parts)
            formatted_messages.append(msg_dict)
        else:
            msg_dict["content"] = str(content)
            formatted_messages.append(msg_dict)

    return formatted_messages, images


# Backwards compatibility alias
_extract_text_and_images = extract_text_and_images
