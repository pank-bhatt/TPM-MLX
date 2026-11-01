"""FastAPI routes for 3D Generation endpoints."""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from tpm_mlx import state
from tpm_mlx.schemas.threed import (
    ThreeDGenerationRequest,
    ThreeDGenerationResponse,
    ThreeDTaskStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/3d", tags=["3D Generation"])

# Thread-safe in-memory task registry for asynchronous 3D synthesis jobs
TASKS: Dict[str, ThreeDTaskStatus] = {}


async def _process_3d_task(task_id: str, req: ThreeDGenerationRequest):
    """Executes long-running 3D mesh and PBR generation on dedicated MLX worker."""
    task = TASKS.get(task_id)
    if not task:
        return

    try:
        task.status = "processing"
        task.stage = "loading_model"
        logger.info(f"Starting async 3D generation task {task_id} for prompt='{req.prompt}'")

        eng = await state.get_or_load_3d_engine(req.model)
        loop = asyncio.get_running_loop()

        def _run():
            with state.active_inference("3d"):
                resp, _ = eng.generate_mesh(req)
                return resp

        task.stage = "synthesizing_3d"
        response = await loop.run_in_executor(state.mlx_executor, _run)

        task.status = "completed"
        task.stage = "completed"
        task.data = response.data
        task.tpm_metrics = response.tpm_metrics
        logger.info(f"Async 3D task {task_id} completed successfully.")
    except Exception as e:
        logger.error(f"Async 3D task {task_id} failed: {e}", exc_info=True)
        task.status = "failed"
        task.stage = "error"
        task.error = str(e)


@router.post("/generations")
async def generate_3d(
    req: ThreeDGenerationRequest,
    async_mode: Optional[bool] = Query(None, alias="async"),
):
    """Generates a watertight 3D mesh (.glb / .usdz) from an image or text prompt.

    If async_mode is True (via JSON body or query param ?async=true), returns a task ID
    immediately with HTTP 202 Accepted. Clients can poll GET /v1/3d/tasks/{task_id}
    at regular intervals (e.g. every 10 seconds).
    """
    is_async = req.async_mode or (async_mode is True)
    if is_async:
        timestamp = int(time.time())
        task_id = f"task_3d_{timestamp}_{uuid.uuid4().hex[:8]}"
        task_status = ThreeDTaskStatus(
            task_id=task_id,
            status="processing",
            stage="queued",
            created=timestamp,
            prompt=req.prompt,
        )
        TASKS[task_id] = task_status
        asyncio.create_task(_process_3d_task(task_id, req))
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=task_status.model_dump(),
        )

    # Synchronous execution for backward compatibility
    try:
        eng = await state.get_or_load_3d_engine(req.model)
        loop = asyncio.get_running_loop()

        def _run_generation():
            with state.active_inference("3d"):
                resp, _ = eng.generate_mesh(req)
                return resp

        response = await loop.run_in_executor(state.mlx_executor, _run_generation)
        return response
    except FileNotFoundError as fnf:
        logger.error(f"3D image input not found: {fnf}")
        raise HTTPException(status_code=400, detail=str(fnf))
    except ValueError as ve:
        logger.error(f"3D validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"3D generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"3D generation failed: {str(e)}")


@router.post("/generations/async", status_code=status.HTTP_202_ACCEPTED)
async def generate_3d_async(req: ThreeDGenerationRequest):
    """Explicit endpoint to enqueue an asynchronous 3D generation job."""
    req.async_mode = True
    return await generate_3d(req, async_mode=True)


@router.get("/tasks/{task_id}", response_model=ThreeDTaskStatus)
async def get_3d_task(task_id: str) -> ThreeDTaskStatus:
    """Returns the current status, stage, and output assets of an async 3D generation task."""
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"3D generation task '{task_id}' not found")
    return task


@router.get("/tasks", response_model=List[ThreeDTaskStatus])
async def list_3d_tasks() -> List[ThreeDTaskStatus]:
    """Lists recent 3D generation tasks."""
    return list(TASKS.values())
