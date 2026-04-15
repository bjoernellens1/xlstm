"""FastAPI application factory for xLSTM Chat.

Creates and configures the FastAPI application with all routes, middleware,
static file serving, and lifecycle management.

The application serves:
    - REST API at ``/api/v1/``
    - WebSocket at ``/api/v1/chat/ws``
    - Interactive API docs at ``/docs`` (Swagger UI)
    - Alternative API docs at ``/redoc`` (ReDoc)
    - Web chat interface at ``/``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chat import __version__
from chat.api.dependencies import init_dependencies
from chat.api.routes import chat, health, models
from chat.core.config import AppConfig, load_config

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Sets up middleware, routes, static files, and the application
    lifecycle (startup/shutdown hooks).

    Args:
        config: Optional application config. If ``None``, loads from
                environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    if config is None:
        config = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifecycle manager.

        On startup: initializes dependencies and optionally auto-loads
        the configured model.
        On shutdown: cleans up resources.
        """
        logger.info("Starting xLSTM Chat v%s", __version__)
        init_dependencies(config)
        logger.info("Dependencies initialized. Ready to load a model.")
        yield
        logger.info("Shutting down xLSTM Chat.")

    app = FastAPI(
        title="xLSTM Chat API",
        description=(
            "A high-performance chat interface and API for xLSTM language models.\n\n"
            "## Features\n"
            "- **Chat Completion**: REST and WebSocket endpoints for text generation\n"
            "- **Model Management**: Switch between small and large xLSTM variants\n"
            "- **Streaming**: Real-time token-by-token streaming via SSE and WebSocket\n"
            "- **Configuration**: Runtime-adjustable generation parameters\n\n"
            "## Model Variants\n"
            "- **small**: xLSTM NeurIPS model — lightweight, CPU-friendly\n"
            "- **large**: xLSTM Large 7B — high-quality, GPU recommended\n"
            "- **vision** *(planned)*: Vision-LSTM — xLSTM as Generic Vision Backbone\n"
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)

    # Static files and frontend
    if FRONTEND_DIR.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(FRONTEND_DIR / "static")),
            name="static",
        )

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            """Serve the chat web interface."""
            return FileResponse(
                str(FRONTEND_DIR / "templates" / "index.html")
            )

    return app
