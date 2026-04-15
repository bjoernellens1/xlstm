"""Health check and system status API routes.

Endpoints:
    - ``GET /api/v1/health`` — Application health check.
    - ``GET /api/v1/config/generation`` — Get generation configuration.
    - ``PATCH /api/v1/config/generation`` — Update generation configuration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from chat import __version__
from chat.api.dependencies import get_app_config, get_model_manager
from chat.api.schemas import (
    GenerationConfigResponse,
    GenerationConfigUpdate,
    HealthResponse,
)
from chat.core.config import AppConfig
from chat.core.model_manager import ModelManager

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the service, including "
    "whether a model is loaded and the application version.",
)
async def health_check(
    manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """Check application health and readiness.

    Returns service status, model readiness, and version information.

    Args:
        manager: Injected model manager.

    Returns:
        Health status response.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=manager.is_loaded(),
        version=__version__,
    )


@router.get(
    "/config/generation",
    response_model=GenerationConfigResponse,
    summary="Get generation config",
    description="Returns the current text generation configuration parameters.",
)
async def get_generation_config(
    config: AppConfig = Depends(get_app_config),
) -> GenerationConfigResponse:
    """Get current generation configuration.

    Args:
        config: Injected application configuration.

    Returns:
        Current generation settings.
    """
    gen = config.generation
    return GenerationConfigResponse(
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
    )


@router.patch(
    "/config/generation",
    response_model=GenerationConfigResponse,
    summary="Update generation config",
    description="Partially update generation configuration. Only provided "
    "fields will be changed.",
)
async def update_generation_config(
    update: GenerationConfigUpdate,
    config: AppConfig = Depends(get_app_config),
) -> GenerationConfigResponse:
    """Update generation configuration at runtime.

    Only fields included in the request body are modified. This allows
    fine-tuning generation behavior without restarting the server.

    Args:
        update: Partial configuration update.
        config: Injected application configuration.

    Returns:
        Updated generation settings.
    """
    gen = config.generation
    if update.max_new_tokens is not None:
        gen.max_new_tokens = update.max_new_tokens
    if update.temperature is not None:
        gen.temperature = update.temperature
    if update.top_k is not None:
        gen.top_k = update.top_k
    if update.top_p is not None:
        gen.top_p = update.top_p
    if update.repetition_penalty is not None:
        gen.repetition_penalty = update.repetition_penalty

    return GenerationConfigResponse(
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.repetition_penalty,
    )
