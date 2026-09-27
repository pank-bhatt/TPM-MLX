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
        role = getattr(m, "role", None) or m.get("role") if isinstance(m, dict) else "user"
        content = getattr(m, "content", None) or m.get("content") if isinstance(m, dict) else m

        if isinstance(content, str):
            formatted_messages.append({"role": role, "content": content})
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
                                if img_url.startswith("data:image"):
                                    header, base64_data = img_url.split(",", 1)
                                    img_bytes = base64.b64decode(base64_data)
                                    img = Image.open(io.BytesIO(img_bytes))
                                    images.append(img)
                                elif img_url.startswith("http"):
                                    import requests
                                    resp = requests.get(img_url, timeout=10)
                                    img = Image.open(io.BytesIO(resp.content))
                                    images.append(img)
                            except Exception as ex:
                                logger.warning(f"Failed to load image from {img_url[:30]}...: {ex}")
            formatted_messages.append({"role": role, "content": " ".join(text_parts)})
        else:
            formatted_messages.append({"role": role, "content": str(content)})

    return formatted_messages, images


# Backwards compatibility alias
_extract_text_and_images = extract_text_and_images
