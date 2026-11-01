"""Pydantic schemas for 3D Generation endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ThreeDGenerationRequest(BaseModel):
    """Request payload for POST /v1/3d/generations."""

    model: Optional[str] = Field(
        default="dgrauet/hunyuan3d-2.1-mlx",
        description="Target 3D generator model identifier",
    )
    image: Optional[str] = Field(
        default=None,
        description="Reference image URL, local path, or base64 data URI",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional text prompt (if generating 2D image first or for conditioning)",
    )
    steps: int = Field(
        default=30,
        ge=1,
        le=50,
        description="Denoising steps (30-50 for base Hunyuan3D-2.1)",
    )
    guidance_scale: Optional[float] = Field(
        default=5.0,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (5.0 for Hunyuan3D-2.1)",
    )
    octree_resolution: int = Field(
        default=256,
        ge=64,
        le=512,
        description="Marching cubes grid resolution (64, 128, 256, 384, 512)",
    )
    target_face_count: int = Field(
        default=40000,
        ge=1000,
        le=300000,
        description="Target polygon triangle face count after decimation",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Deterministic random seed",
    )
    auto_remove_bg: bool = Field(
        default=True,
        description="Whether to automatically isolate the foreground subject via rembg",
    )
    pbr_paint: bool = Field(
        default=True,
        description="Whether to execute Stage 2 multi-view PBR texture synthesis",
    )
    paint_resolution: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Stage 2 multi-view diffusion render resolution (512, 768, 1024)",
    )
    texture_resolution: int = Field(
        default=2048,
        description="Output UV texture map resolution (e.g. 1024, 2048, 4096)",
    )
    export_formats: List[str] = Field(
        default_factory=lambda: ["glb"],
        description="Target 3D formats to export ('glb', 'usdz', 'obj')",
    )
    response_format: str = Field(
        default="url",
        description="Response format ('url' or 'b64_json')",
    )
    async_mode: bool = Field(
        default=False,
        description="Whether to execute asynchronously and return a task ID immediately",
    )


class ThreeDMetrics(BaseModel):
    """Performance telemetry and geometry metadata."""

    generation_time_s: float
    faces: int
    vertices: int
    peak_memory_gb: float
    has_pbr_texture: bool = False
    octree_resolution: int = 256
    timing_breakdown: Optional[Dict[str, float]] = None


class ThreeDModelItem(BaseModel):
    """File links or encoded data for generated 3D meshes."""

    glb_url: Optional[str] = None
    usdz_url: Optional[str] = None
    obj_url: Optional[str] = None
    b64_json: Optional[str] = None


class ThreeDGenerationResponse(BaseModel):
    """Response payload for POST /v1/3d/generations."""

    created: int
    data: List[ThreeDModelItem]
    tpm_metrics: ThreeDMetrics


class ThreeDTaskStatus(BaseModel):
    """Status and telemetry for asynchronous 3D generation jobs."""

    task_id: str
    status: str = Field(..., description="Job status: 'pending', 'processing', 'completed', 'failed'")
    stage: Optional[str] = Field(default=None, description="Current pipeline stage")
    created: int
    prompt: Optional[str] = None
    data: Optional[List[ThreeDModelItem]] = None
    tpm_metrics: Optional[ThreeDMetrics] = None
    error: Optional[str] = None

