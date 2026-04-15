"""Chat API routes for text generation.

Provides both REST and WebSocket endpoints for chat interactions.

Endpoints:
    - ``POST /api/v1/chat`` — Generate a chat response (batch or streaming).
    - ``WS /api/v1/chat/ws`` — WebSocket endpoint for real-time streaming chat.

The streaming REST endpoint uses Server-Sent Events (SSE) for compatibility
with standard HTTP clients. The WebSocket endpoint provides lower-latency
bidirectional communication for the web frontend.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from chat.api.dependencies import get_inference_engine, get_model_manager
from chat.api.schemas import ChatMessage, ChatRequest, ChatResponse
from chat.core.inference import InferenceEngine
from chat.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])


def _build_prompt(messages: list[ChatMessage]) -> str:
    """Convert a list of chat messages into a single prompt string.

    Uses a simple format: each message is prefixed with its role,
    and the assistant turn is left open for generation.

    Args:
        messages: List of chat messages.

    Returns:
        Formatted prompt string.
    """
    parts = []
    for msg in messages:
        parts.append(f"{msg.role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Generate a chat response",
    description="Send a conversation and receive a generated response. "
    "Set `stream=true` for Server-Sent Events streaming.",
    responses={
        200: {"description": "Generated chat response"},
        503: {"description": "Model not loaded"},
    },
)
async def chat_completion(
    request: ChatRequest,
    engine: InferenceEngine = Depends(get_inference_engine),
    manager: ModelManager = Depends(get_model_manager),
) -> ChatResponse | StreamingResponse:
    """Generate a chat completion from the conversation history.

    If ``request.stream`` is ``True``, returns a ``text/event-stream`` response
    with tokens sent as they are generated.

    Args:
        request: The chat request with messages and generation parameters.
        engine: Injected inference engine.
        manager: Injected model manager.

    Returns:
        ChatResponse for batch mode, or StreamingResponse for stream mode.

    Raises:
        HTTPException: If the model is not loaded (503).
    """
    if not manager.is_loaded():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503, detail="Model not loaded. Load a model first."
        )

    prompt = _build_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_response(engine, prompt, request),
            media_type="text/event-stream",
        )

    result = await engine.generate(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )

    return ChatResponse(
        message=ChatMessage(role="assistant", content=result["text"]),
        usage={
            "tokens_generated": result["tokens_generated"],
            "generation_time_s": result["generation_time_s"],
            "tokens_per_second": result["tokens_per_second"],
        },
    )


async def _stream_response(
    engine: InferenceEngine,
    prompt: str,
    request: ChatRequest,
):
    """Async generator that yields SSE-formatted token events.

    Each event contains a JSON payload with the generated token fragment.
    A final ``[DONE]`` event signals the end of generation.

    Args:
        engine: The inference engine.
        prompt: The formatted prompt string.
        request: The original chat request with generation parameters.

    Yields:
        SSE-formatted string events.
    """
    async for token in engine.generate_stream(
        prompt=prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    ):
        data = json.dumps({"token": token})
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


@router.websocket("/chat/ws")
async def chat_websocket(
    websocket: WebSocket,
    engine: InferenceEngine = Depends(get_inference_engine),
    manager: ModelManager = Depends(get_model_manager),
) -> None:
    """WebSocket endpoint for real-time streaming chat.

    Protocol:
        1. Client connects.
        2. Client sends JSON: ``{"messages": [...], "max_new_tokens": N, ...}``
        3. Server streams back JSON: ``{"token": "..."}`` per generated token.
        4. Server sends ``{"done": true}`` when generation is complete.
        5. Client can send another message or close the connection.

    Args:
        websocket: The WebSocket connection.
        engine: Injected inference engine.
        manager: Injected model manager.
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")

    try:
        while True:
            data = await websocket.receive_json()

            if not manager.is_loaded():
                await websocket.send_json(
                    {"error": "Model not loaded. Load a model first."}
                )
                continue

            messages = [ChatMessage(**m) for m in data.get("messages", [])]
            if not messages:
                await websocket.send_json({"error": "No messages provided."})
                continue

            prompt = _build_prompt(messages)

            async for token in engine.generate_stream(
                prompt=prompt,
                max_new_tokens=data.get("max_new_tokens"),
                temperature=data.get("temperature"),
                top_k=data.get("top_k"),
            ):
                await websocket.send_json({"token": token})

            await websocket.send_json({"done": True})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
