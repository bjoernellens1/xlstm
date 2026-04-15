"""FastAPI dependency injection for shared application state.

Provides access to the model manager and inference engine instances
via FastAPI's dependency injection system. All route handlers should
use these dependencies rather than importing globals directly.

Usage in route handlers::

    @router.post("/chat")
    async def chat(
        manager: ModelManager = Depends(get_model_manager),
        engine: InferenceEngine = Depends(get_inference_engine),
    ):
        ...
"""

from __future__ import annotations

from chat.core.config import AppConfig
from chat.core.inference import InferenceEngine
from chat.core.model_manager import ModelManager

# Module-level singletons, initialized by the app startup hook.
_model_manager: ModelManager | None = None
_inference_engine: InferenceEngine | None = None
_app_config: AppConfig | None = None


def init_dependencies(config: AppConfig) -> None:
    """Initialize shared dependencies from application config.

    Called once during application startup. Creates the model manager
    and inference engine singletons.

    Args:
        config: The application configuration.
    """
    global _model_manager, _inference_engine, _app_config
    _app_config = config
    _model_manager = ModelManager(config)
    _inference_engine = InferenceEngine(_model_manager, config.generation)


def get_model_manager() -> ModelManager:
    """FastAPI dependency: get the model manager singleton.

    Returns:
        The active :class:`ModelManager` instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _model_manager is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _model_manager


def get_inference_engine() -> InferenceEngine:
    """FastAPI dependency: get the inference engine singleton.

    Returns:
        The active :class:`InferenceEngine` instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _inference_engine is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _inference_engine


def get_app_config() -> AppConfig:
    """FastAPI dependency: get the application configuration.

    Returns:
        The active :class:`AppConfig` instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _app_config is None:
        raise RuntimeError("Dependencies not initialized. Call init_dependencies() first.")
    return _app_config
