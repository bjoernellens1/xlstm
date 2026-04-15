"""Entry point for running xLSTM Chat as a module.

Usage:
    python -m chat                      # Start with default settings
    python -m chat --port 8080          # Custom port
    python -m chat --variant large      # Load large model variant

All configuration can also be set via environment variables
(see ``chat.core.config`` for details).
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from chat.api.app import create_app
from chat.core.config import DeviceType, ModelVariant, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="xLSTM Chat — Chat GUI and API for xLSTM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m chat                          # Start with defaults\n"
            "  python -m chat --port 8080              # Custom port\n"
            "  python -m chat --variant large           # Use large model\n"
            "  python -m chat --device cpu              # Force CPU\n"
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["small", "large"],
        default=None,
        help="Model variant to load on startup",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for large variant)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of uvicorn workers (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the xLSTM Chat application."""
    args = parse_args()
    config = load_config()

    # Override config from CLI args
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.variant:
        config.model.variant = ModelVariant(args.variant)
    if args.device:
        config.model.device = DeviceType(args.device)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.workers:
        config.server.workers = args.workers

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
