"""Model management API routes.

Endpoints for querying model information, switching between model variants,
and listing available models.

Endpoints:
    - ``GET /api/v1/models`` — List available model variants.
    - ``GET /api/v1/models/current`` — Get info about the loaded model.
    - ``POST /api/v1/models/load`` — Load or switch to a specific model variant.
    - ``POST /api/v1/models/unload`` — Unload the current model.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from chat.api.dependencies import get_model_manager
from chat.api.schemas import (
    ModelInfoResponse,
    ModelSwitchRequest,
    ModelVariantInfo,
)
from chat.core.config import ModelVariant
from chat.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get(
    "",
    response_model=list[ModelVariantInfo],
    summary="List available model variants",
    description="Returns all available xLSTM model variants including their "
    "status (available or planned).",
)
async def list_models(
    manager: ModelManager = Depends(get_model_manager),
) -> list[ModelVariantInfo]:
    """List all model variants that can be loaded.

    Includes both currently available variants and planned future ones
    (e.g., Vision-LSTM).

    Args:
        manager: Injected model manager.

    Returns:
        List of model variant information objects.
    """
    variants = manager.available_variants()
    return [ModelVariantInfo(**v) for v in variants]


@router.get(
    "/current",
    response_model=ModelInfoResponse,
    summary="Get current model info",
    description="Returns detailed information about the currently loaded model, "
    "including parameter count, device, and capabilities.",
)
async def get_current_model(
    manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Get metadata about the currently loaded model.

    Returns comprehensive information including parameter counts, device
    placement, and capability flags.

    Args:
        manager: Injected model manager.

    Returns:
        Model information response.
    """
    info = manager.model_info()
    return ModelInfoResponse(**info)


@router.post(
    "/load",
    response_model=ModelInfoResponse,
    summary="Load a model variant",
    description="Load or switch to a specific xLSTM model variant. "
    "If a model is already loaded, it will be unloaded first.",
    responses={
        200: {"description": "Model loaded successfully"},
        400: {"description": "Invalid variant or loading failed"},
    },
)
async def load_model(
    request: ModelSwitchRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Load or switch to a specific model variant.

    Unloads any currently loaded model before loading the new one.
    This operation may take significant time for large models.

    Args:
        request: The model switch request specifying the variant.
        manager: Injected model manager.

    Returns:
        Updated model information after loading.

    Raises:
        HTTPException: If loading fails (400).
    """
    try:
        variant = ModelVariant(request.variant.value)
        await manager.load_model(variant)
        info = manager.model_info()
        return ModelInfoResponse(**info)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/unload",
    summary="Unload the current model",
    description="Unload the current model and free GPU/CPU memory.",
    responses={200: {"description": "Model unloaded successfully"}},
)
async def unload_model(
    manager: ModelManager = Depends(get_model_manager),
) -> dict:
    """Unload the current model and free resources.

    After unloading, inference endpoints will return 503 until a new
    model is loaded.

    Args:
        manager: Injected model manager.

    Returns:
        Confirmation message.
    """
    manager.unload_model()
    return {"message": "Model unloaded successfully."}
