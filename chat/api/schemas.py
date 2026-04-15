"""Pydantic schemas for API request/response validation.

These schemas define the data contracts for the REST and WebSocket APIs.
They are also used to auto-generate the OpenAPI documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Chat Schemas ─────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single chat message.

    Attributes:
        role: The message author role (``user`` or ``assistant``).
        content: The text content of the message.
    """

    role: str = Field(
        ...,
        description="Message author: 'user' or 'assistant'",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="The text content of the message",
        examples=["Hello, how are you?"],
    )


class ChatRequest(BaseModel):
    """Request body for the chat completion endpoint.

    Attributes:
        messages: List of conversation messages.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy, higher = more random).
        top_k: Top-k sampling parameter.
        stream: Whether to stream the response token by token.
    """

    messages: list[ChatMessage] = Field(
        ...,
        description="Conversation history as a list of messages",
        min_length=1,
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_k: Optional[int] = Field(
        None,
        ge=0,
        description="Top-k sampling parameter",
    )
    stream: bool = Field(
        False,
        description="Enable streaming response",
    )


class ChatResponse(BaseModel):
    """Response body for the chat completion endpoint.

    Attributes:
        message: The generated assistant response.
        usage: Token usage statistics.
    """

    message: ChatMessage
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage and performance statistics",
    )


# ── Model Schemas ────────────────────────────────────────────────────────────


class ModelVariantEnum(str, Enum):
    """Available model variants for the switch endpoint."""

    small = "small"
    large = "large"


class ModelSwitchRequest(BaseModel):
    """Request to switch the active model variant.

    Attributes:
        variant: The model variant to load.
    """

    variant: ModelVariantEnum = Field(
        ..., description="Model variant to load"
    )


class ModelInfoResponse(BaseModel):
    """Response with information about the currently loaded model.

    Attributes:
        loaded: Whether a model is currently loaded.
        variant: The active model variant name.
        device: The compute device in use.
        parameter_count: Total number of model parameters.
        parameter_count_human: Human-readable parameter count.
        vocab_size: Model vocabulary size.
        max_context_length: Maximum context window length.
        has_tokenizer: Whether a tokenizer is available.
        supports_vision: Whether Vision-LSTM is supported.
    """

    loaded: bool
    variant: Optional[str] = None
    device: Optional[str] = None
    parameter_count: Optional[int] = None
    parameter_count_human: Optional[str] = None
    vocab_size: Optional[int] = None
    max_context_length: Optional[int] = None
    has_tokenizer: Optional[bool] = None
    supports_vision: bool = False


class ModelVariantInfo(BaseModel):
    """Information about a single model variant.

    Attributes:
        name: Variant identifier.
        description: Human-readable description.
        status: Availability status (``available`` or ``planned``).
    """

    name: str
    description: str
    status: str


# ── Health Schemas ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service health status string.
        model_loaded: Whether a model is loaded and ready.
        version: Application version string.
    """

    status: str = "healthy"
    model_loaded: bool = False
    version: str = "0.1.0"


# ── Generation Config Schemas ────────────────────────────────────────────────


class GenerationConfigResponse(BaseModel):
    """Current generation configuration values.

    Attributes:
        max_new_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        top_k: Top-k sampling value.
        top_p: Nucleus sampling value.
        repetition_penalty: Repetition penalty factor.
    """

    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float


class GenerationConfigUpdate(BaseModel):
    """Partial update to generation configuration.

    All fields are optional; only provided fields will be updated.

    Attributes:
        max_new_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        top_k: Top-k sampling value.
        top_p: Nucleus sampling value.
        repetition_penalty: Repetition penalty factor.
    """

    max_new_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=3.0)
