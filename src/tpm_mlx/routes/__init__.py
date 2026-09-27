# Copyright © 2026 TPM-MLX Authors. All rights reserved.

from tpm_mlx.routes.views import router as views_router
from tpm_mlx.routes.models import router as models_router
from tpm_mlx.routes.chat import router as chat_router
from tpm_mlx.routes.images import router as images_router

__all__ = [
    "views_router",
    "models_router",
    "chat_router",
    "images_router",
]
