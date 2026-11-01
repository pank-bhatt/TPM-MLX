"""Image preprocessor for Hunyuan3D-2.1.

Handles background removal (rembg), alpha bounding-box cropping,
aspect-preserving centering with border padding, white background compositing,
and conversion to MLX channels-last tensor format.
"""

import logging
from typing import Any, Optional, Tuple, Union
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def remove_background(image: Image.Image) -> Image.Image:
    """Removes image background using rembg if available.

    Falls back to original image if rembg fails or is unavailable.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("onnxruntime") is None:
            logger.warning("onnxruntime is not installed; skipping rembg background removal.")
            return image.convert("RGBA") if image.mode != "RGBA" else image

        import rembg
        return rembg.remove(image)
    except (Exception, SystemExit, BaseException) as e:
        logger.warning(f"rembg background removal failed or not installed ({e}); using input image directly.")
        return image.convert("RGBA") if image.mode != "RGBA" else image


class ImageProcessorV2:
    """Preprocesses reference images for 3D generation.

    Attributes:
        size: Target square image dimension (default 518 for DINOv2).
        border_ratio: Fraction of canvas reserved for empty border around object (default 0.15).
    """

    def __init__(self, size: int = 518, border_ratio: float = 0.15):
        self.size = size
        self.border_ratio = border_ratio

    def recenter_and_composite(
        self,
        image: Image.Image,
        border_ratio: Optional[float] = None,
        return_rgba: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """Recenters foreground object using alpha channel and composites onto white canvas."""
        ratio = border_ratio if border_ratio is not None else self.border_ratio
        image_rgba = image.convert("RGBA")
        np_img = np.asarray(image_rgba)

        alpha = np_img[..., 3]
        coords = np.nonzero(alpha > 15)

        if len(coords[0]) == 0 or len(coords[1]) == 0:
            # Empty image or completely transparent, fallback to RGB white
            bg = Image.new("RGB", (self.size, self.size), (255, 255, 255))
            if return_rgba:
                return bg, Image.new("RGBA", (self.size, self.size), (255, 255, 255, 0))
            return bg

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())

        h = y_max - y_min + 1
        w = x_max - x_min + 1

        # Crop foreground object
        cropped = image_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))

        # Calculate scaled dimensions
        target_span = int(self.size * (1.0 - ratio))
        scale = max(1e-4, target_span / max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cropped.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Place centered onto a pure white canvas
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        paste_x = (self.size - new_w) // 2
        paste_y = (self.size - new_h) // 2

        canvas.paste(resized, (paste_x, paste_y), mask=resized.split()[3])

        if return_rgba:
            canvas_rgba = Image.new("RGBA", (self.size, self.size), (255, 255, 255, 0))
            canvas_rgba.paste(resized, (paste_x, paste_y), mask=resized.split()[3])
            return canvas, canvas_rgba

        return canvas

    def preprocess(
        self,
        image: Union[str, Path, Image.Image, mx.array],
        auto_remove_bg: bool = True,
        border_ratio: Optional[float] = None,
        dtype: mx.Dtype = mx.float16,
        return_rgba: bool = False,
    ) -> Tuple[Any, ...]:
        """Preprocesses input image into a normalized channels-last MLX array.

        Args:
            image: Path to image, PIL Image, or existing tensor.
            auto_remove_bg: Whether to run background removal.
            border_ratio: Fraction of border padding.
            dtype: Output MLX tensor dtype.

        Returns:
            Tuple of (mlx_array (1, H, W, 3) in range [-1, 1], composite_pil_image).
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image))
        elif isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, mx.array):
            # Already a tensor
            return image, None
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Background removal if requested and image is not already RGBA with distinct transparent alpha
        if auto_remove_bg:
            needs_rembg = True
            if pil_img.mode == "RGBA":
                np_a = np.asarray(pil_img)[..., 3]
                # If image already has an isolated transparent background (>5% pixels transparent)
                if float(np.mean(np_a < 30)) > 0.05:
                    needs_rembg = False
            if needs_rembg:
                pil_img = remove_background(pil_img)

        # Recenter foreground and composite onto white canvas
        if return_rgba:
            composited, composited_rgba = self.recenter_and_composite(
                pil_img, border_ratio=border_ratio, return_rgba=True
            )
        else:
            composited = self.recenter_and_composite(
                pil_img, border_ratio=border_ratio, return_rgba=False
            )
            composited_rgba = None

        # Convert to numpy array normalized to [-1, 1]
        arr = np.array(composited, dtype=np.float32) / 127.5 - 1.0  # (H, W, 3)
        tensor = mx.array(arr[None], dtype=dtype)  # (1, H, W, 3)
        if return_rgba:
            return tensor, composited, composited_rgba
        return tensor, composited
