"""Application configuration for xLSTM Chat.

Centralizes all configuration with environment variable support and sensible defaults.
All settings can be overridden via environment variables prefixed with ``XLSTM_CHAT_``.

Example:
    ``XLSTM_CHAT_HOST=0.0.0.0 XLSTM_CHAT_PORT=8080 python -m chat``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelVariant(str, Enum):
    """Available xLSTM model variants.

    Attributes:
        SMALL: NeurIPS xLSTM model (lightweight, fast).
        LARGE: xLSTM Large / 7B model (high quality, requires GPU).
    """

    SMALL = "small"
    LARGE = "large"


class DeviceType(str, Enum):
    """Compute device options.

    Attributes:
        CPU: Run on CPU (slower but always available).
        CUDA: Run on NVIDIA GPU.
        AUTO: Automatically select the best available device.
    """

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


@dataclass
class ModelConfig:
    """Configuration for xLSTM model parameters.

    Attributes:
        variant: Which model variant to load (``small`` or ``large``).
        checkpoint_path: Path to pretrained model checkpoint (required for ``large``).
        device: Compute device to use.
        dtype: Data type for model weights (``float32``, ``float16``, ``bfloat16``).
        max_context_length: Maximum sequence length the model can process.
        embedding_dim: Embedding dimension (used for small variant).
        num_heads: Number of attention heads.
        num_blocks: Number of transformer/xLSTM blocks.
        vocab_size: Vocabulary size.
    """

    variant: ModelVariant = ModelVariant.SMALL
    checkpoint_path: Optional[str] = None
    device: DeviceType = DeviceType.AUTO
    dtype: str = "float32"
    max_context_length: int = 256
    # Small model defaults
    embedding_dim: int = 128
    num_heads: int = 4
    num_blocks: int = 7
    vocab_size: int = 50304


@dataclass
class GenerationConfig:
    """Configuration for text generation behavior.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate per response.
        temperature: Sampling temperature (higher = more random). Set 0 for greedy.
        top_k: Top-k sampling parameter. 0 disables.
        top_p: Nucleus sampling parameter. 1.0 disables.
        repetition_penalty: Penalty for repeated tokens. 1.0 disables.
    """

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass
class ServerConfig:
    """Web server configuration.

    Attributes:
        host: Bind address for the server.
        port: Port number for the server.
        workers: Number of uvicorn workers.
        cors_origins: Allowed CORS origins (comma-separated or list).
        log_level: Logging level.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"


@dataclass
class AppConfig:
    """Top-level application configuration.

    Combines model, generation, and server settings. Values can be overridden
    via environment variables prefixed with ``XLSTM_CHAT_``.

    Attributes:
        model: Model-related settings.
        generation: Text generation settings.
        server: Web server settings.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Environment variables are mapped to config fields using the ``XLSTM_CHAT_`` prefix.
    For example, ``XLSTM_CHAT_PORT=8080`` sets ``server.port`` to ``8080``.

    Returns:
        Fully populated :class:`AppConfig` instance.
    """
    config = AppConfig()

    env_map = {
        "XLSTM_CHAT_MODEL_VARIANT": ("model", "variant", ModelVariant),
        "XLSTM_CHAT_CHECKPOINT_PATH": ("model", "checkpoint_path", str),
        "XLSTM_CHAT_DEVICE": ("model", "device", DeviceType),
        "XLSTM_CHAT_DTYPE": ("model", "dtype", str),
        "XLSTM_CHAT_MAX_CONTEXT_LENGTH": ("model", "max_context_length", int),
        "XLSTM_CHAT_EMBEDDING_DIM": ("model", "embedding_dim", int),
        "XLSTM_CHAT_NUM_HEADS": ("model", "num_heads", int),
        "XLSTM_CHAT_NUM_BLOCKS": ("model", "num_blocks", int),
        "XLSTM_CHAT_VOCAB_SIZE": ("model", "vocab_size", int),
        "XLSTM_CHAT_MAX_NEW_TOKENS": ("generation", "max_new_tokens", int),
        "XLSTM_CHAT_TEMPERATURE": ("generation", "temperature", float),
        "XLSTM_CHAT_TOP_K": ("generation", "top_k", int),
        "XLSTM_CHAT_TOP_P": ("generation", "top_p", float),
        "XLSTM_CHAT_REPETITION_PENALTY": ("generation", "repetition_penalty", float),
        "XLSTM_CHAT_HOST": ("server", "host", str),
        "XLSTM_CHAT_PORT": ("server", "port", int),
        "XLSTM_CHAT_WORKERS": ("server", "workers", int),
        "XLSTM_CHAT_LOG_LEVEL": ("server", "log_level", str),
    }

    for env_var, (section, attr, typ) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            section_obj = getattr(config, section)
            setattr(section_obj, attr, typ(value))

    cors = os.environ.get("XLSTM_CHAT_CORS_ORIGINS")
    if cors is not None:
        config.server.cors_origins = [o.strip() for o in cors.split(",")]

    return config
