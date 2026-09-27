# Copyright © 2026 TPM-MLX Authors. All rights reserved.

"""
HTML View and Static Playground routes.
"""

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from tpm_mlx import state

router = APIRouter(tags=["views"])


@router.get("/", response_class=HTMLResponse)
async def serve_playground():
    """Serves the static Web Playground HTML."""
    playground_path = state.static_dir / "playground.html"

    if not playground_path.exists():
        return HTMLResponse(
            content="<h3>Playground HTML not found. Run building steps.</h3>",
            status_code=404,
        )

    with open(playground_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)
