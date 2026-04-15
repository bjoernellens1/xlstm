"""Tests for core configuration module."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from chat.core.config import (
    AppConfig,
    DeviceType,
    GenerationConfig,
    ModelConfig,
    ModelVariant,
    ServerConfig,
    load_config,
)


class TestModelVariant:
    """Tests for ModelVariant enum."""

    def test_variants(self) -> None:
        assert ModelVariant.SMALL.value == "small"
        assert ModelVariant.LARGE.value == "large"


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_devices(self) -> None:
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.AUTO.value == "auto"


class TestModelConfig:
    """Tests for ModelConfig defaults."""

    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.variant == ModelVariant.SMALL
        assert cfg.device == DeviceType.AUTO
        assert cfg.vocab_size == 50304
        assert cfg.embedding_dim == 128
        assert cfg.num_heads == 4
        assert cfg.num_blocks == 7


class TestGenerationConfig:
    """Tests for GenerationConfig defaults."""

    def test_defaults(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.7
        assert cfg.top_k == 50
        assert cfg.top_p == 0.9
        assert cfg.repetition_penalty == 1.1


class TestServerConfig:
    """Tests for ServerConfig defaults."""

    def test_defaults(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.workers == 1


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_default_config(self) -> None:
        config = load_config()
        assert isinstance(config, AppConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.server, ServerConfig)

    @mock.patch.dict(os.environ, {"XLSTM_CHAT_PORT": "9090"})
    def test_env_override_port(self) -> None:
        config = load_config()
        assert config.server.port == 9090

    @mock.patch.dict(os.environ, {"XLSTM_CHAT_TEMPERATURE": "1.5"})
    def test_env_override_temperature(self) -> None:
        config = load_config()
        assert config.generation.temperature == 1.5

    @mock.patch.dict(os.environ, {"XLSTM_CHAT_MODEL_VARIANT": "large"})
    def test_env_override_variant(self) -> None:
        config = load_config()
        assert config.model.variant == ModelVariant.LARGE

    @mock.patch.dict(os.environ, {"XLSTM_CHAT_CORS_ORIGINS": "http://localhost:3000,http://example.com"})
    def test_env_override_cors(self) -> None:
        config = load_config()
        assert len(config.server.cors_origins) == 2
        assert "http://localhost:3000" in config.server.cors_origins
