"""Model manager for loading and managing xLSTM model variants.

Provides a singleton-style manager that handles model lifecycle including
loading, device placement, and variant switching. Designed to support both
the small (NeurIPS) and large (7B) xLSTM architectures, with a future
extension point for Vision-LSTM.

Architecture Overview::

    ModelManager
    ├── load_model(variant)  → loads small or large xLSTM
    ├── get_model()          → returns the currently loaded model
    ├── model_info()         → returns metadata about the active model
    └── [future] load_vision_model()  → Vision-LSTM support

Usage:
    >>> manager = ModelManager(config)
    >>> await manager.load_model()
    >>> model = manager.get_model()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Optional

import torch

from .config import AppConfig, DeviceType, ModelVariant

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages xLSTM model loading, configuration, and lifecycle.

    This class is designed as a singleton per application instance. It handles:

    - Loading the correct model variant (small or large)
    - Device placement (CPU/CUDA/auto)
    - Thread-safe model access
    - Model metadata reporting

    Attributes:
        config: Application configuration.
        model: The currently loaded model instance, or ``None``.
        tokenizer: The tokenizer for the loaded model, or ``None``.
        device: The resolved compute device string.
        variant: The currently loaded model variant.

    Example:
        >>> config = AppConfig()
        >>> manager = ModelManager(config)
        >>> await manager.load_model()
        >>> info = manager.model_info()
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Any = None
        self.device: str = self._resolve_device()
        self.variant: Optional[ModelVariant] = None
        self._lock = asyncio.Lock()
        self._loaded = False

    def _resolve_device(self) -> str:
        """Resolve the compute device based on configuration and availability.

        Returns:
            Device string (``"cpu"`` or ``"cuda"``).
        """
        device_cfg = self.config.model.device
        if device_cfg == DeviceType.AUTO:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_cfg.value

    async def load_model(
        self, variant: Optional[ModelVariant] = None
    ) -> None:
        """Load an xLSTM model variant.

        If a model is already loaded, it will be unloaded first. This method
        is thread-safe and can be called concurrently.

        Args:
            variant: Model variant to load. Defaults to the configured variant.

        Raises:
            RuntimeError: If model loading fails.
            FileNotFoundError: If checkpoint path is invalid for the large variant.
        """
        async with self._lock:
            target_variant = variant or self.config.model.variant

            if self._loaded and self.variant == target_variant:
                logger.info("Model %s already loaded, skipping.", target_variant.value)
                return

            self._unload_model()

            logger.info(
                "Loading xLSTM %s model on %s...",
                target_variant.value,
                self.device,
            )

            loop = asyncio.get_event_loop()
            if target_variant == ModelVariant.SMALL:
                await loop.run_in_executor(None, self._load_small_model)
            elif target_variant == ModelVariant.LARGE:
                await loop.run_in_executor(None, self._load_large_model)

            self.variant = target_variant
            self._loaded = True
            logger.info("Model loaded successfully.")

    def _load_small_model(self) -> None:
        """Load the small (NeurIPS) xLSTM language model.

        Uses the ``xLSTMLMModel`` with configurable parameters suitable
        for lightweight inference on CPU or GPU.
        """
        from xlstm import xLSTMLMModel, xLSTMLMModelConfig

        cfg = xLSTMLMModelConfig(
            vocab_size=self.config.model.vocab_size,
            embedding_dim=self.config.model.embedding_dim,
            num_heads=self.config.model.num_heads,
            num_blocks=self.config.model.num_blocks,
            context_length=self.config.model.max_context_length,
            slstm_at=[],
            mlstm_block={
                "mlstm": {
                    "conv1d_kernel_size": 4,
                    "qkv_proj_blocksize": 4,
                    "num_heads": self.config.model.num_heads,
                }
            },
        )

        self.model = xLSTMLMModel(cfg)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._load_tokenizer()

    def _load_large_model(self) -> None:
        """Load the large xLSTM (7B architecture) model.

        Requires a valid checkpoint path in the configuration. Uses the
        ``xLSTMLarge`` model with optimized inference kernels.

        Raises:
            FileNotFoundError: If checkpoint_path is not set or doesn't exist.
        """
        checkpoint = self.config.model.checkpoint_path
        if checkpoint is not None and checkpoint != "":
            from xlstm.xlstm_large.from_pretrained import load_from_pretrained

            kernel_kwargs = {}
            if self.device == "cpu":
                kernel_kwargs = {
                    "chunkwise_kernel_name": "chunkwise--native_autograd",
                    "sequence_kernel_name": "native_sequence__native",
                    "step_kernel_name": "native",
                }

            self.model = load_from_pretrained(
                checkpoint_path=checkpoint,
                backend_mode="inference",
                return_last_states=True,
                **kernel_kwargs,
            )
        else:
            from xlstm.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig

            xlstm_config = xLSTMLargeConfig(
                embedding_dim=self.config.model.embedding_dim,
                num_heads=self.config.model.num_heads,
                num_blocks=self.config.model.num_blocks,
                vocab_size=self.config.model.vocab_size,
                return_last_states=True,
                mode="inference",
                chunkwise_kernel="chunkwise--native_autograd",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
            )
            self.model = xLSTMLarge(xlstm_config)

        self.model = self.model.to(self.device)
        self.model.eval()
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """Load the tokenizer for text encoding/decoding.

        Uses the HuggingFace GPT-2 tokenizer as the default, since xLSTM
        models typically use a BPE-style vocabulary.
        """
        try:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_pretrained("gpt2")
            logger.info("Loaded GPT-2 tokenizer.")
        except Exception:
            logger.warning(
                "Could not load tokenizer. Text encoding/decoding will be unavailable."
            )
            self.tokenizer = None

    def _unload_model(self) -> None:
        """Unload the current model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self.tokenizer = None
            self._loaded = False
            self.variant = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Previous model unloaded.")

    def unload_model(self) -> None:
        """Public interface to unload the current model and free resources.

        Safe to call even when no model is loaded.
        """
        self._unload_model()

    def get_model(self) -> torch.nn.Module:
        """Get the currently loaded model.

        Returns:
            The loaded PyTorch model.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() first."
            )
        return self.model

    def is_loaded(self) -> bool:
        """Check whether a model is currently loaded.

        Returns:
            ``True`` if a model is loaded and ready for inference.
        """
        return self._loaded

    def model_info(self) -> dict[str, Any]:
        """Get metadata about the currently loaded model.

        Returns:
            Dictionary containing model variant, device, parameter count,
            and configuration details.
        """
        if not self._loaded:
            return {"loaded": False, "variant": None}

        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "loaded": True,
            "variant": self.variant.value if self.variant else None,
            "device": self.device,
            "parameter_count": param_count,
            "parameter_count_human": _human_readable_params(param_count),
            "vocab_size": self.config.model.vocab_size,
            "max_context_length": self.config.model.max_context_length,
            "has_tokenizer": self.tokenizer is not None,
            "supports_vision": False,  # Future: Vision-LSTM support
        }

    def available_variants(self) -> list[dict[str, str]]:
        """List all available model variants.

        Returns:
            List of variant info dicts with name, description, and status.
        """
        return [
            {
                "name": ModelVariant.SMALL.value,
                "description": "xLSTM NeurIPS model - lightweight, CPU-friendly",
                "status": "available",
            },
            {
                "name": ModelVariant.LARGE.value,
                "description": "xLSTM Large 7B - high-quality, GPU recommended",
                "status": "available",
            },
            {
                "name": "vision",
                "description": "Vision-LSTM - xLSTM as Generic Vision Backbone",
                "status": "planned",
            },
        ]


def _human_readable_params(count: int) -> str:
    """Convert a parameter count to a human-readable string.

    Args:
        count: Number of parameters.

    Returns:
        Formatted string like ``"7.0B"`` or ``"125.3M"``.
    """
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    if count >= 1e6:
        return f"{count / 1e6:.1f}M"
    if count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)
