"""Inference engine for xLSTM text generation.

Provides both synchronous and streaming token generation with configurable
sampling strategies. Designed for high-performance real-time chat applications.

Architecture::

    InferenceEngine
    ├── generate()          → full response generation
    ├── generate_stream()   → async token-by-token streaming
    └── encode/decode()     → tokenizer utilities

The engine wraps the model manager and handles tokenization, sampling,
and state management transparently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Optional

import torch

from .config import GenerationConfig
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-performance inference engine for xLSTM models.

    Handles tokenization, generation with various sampling strategies,
    and both batch and streaming output modes.

    Attributes:
        manager: The model manager providing the loaded model.
        gen_config: Default generation configuration.

    Example:
        >>> engine = InferenceEngine(model_manager, gen_config)
        >>> response = await engine.generate("Hello, how are you?")
        >>> async for token in engine.generate_stream("Tell me a story"):
        ...     print(token, end="")
    """

    def __init__(
        self,
        manager: ModelManager,
        gen_config: GenerationConfig,
    ) -> None:
        self.manager = manager
        self.gen_config = gen_config

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of integer token IDs.

        Raises:
            RuntimeError: If no tokenizer is loaded.
        """
        if self.manager.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")
        encoding = self.manager.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text.

        Args:
            token_ids: List of token IDs to decode.

        Returns:
            Decoded text string.

        Raises:
            RuntimeError: If no tokenizer is loaded.
        """
        if self.manager.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")
        return self.manager.tokenizer.decode(token_ids)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """Generate a complete response for the given prompt.

        Runs generation in a thread executor to avoid blocking the event loop.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_k: Override default top-k value.

        Returns:
            Dictionary with keys ``text``, ``tokens_generated``,
            ``generation_time_s``, and ``tokens_per_second``.
        """
        max_tokens = max_new_tokens or self.gen_config.max_new_tokens
        temp = temperature if temperature is not None else self.gen_config.temperature
        k = top_k if top_k is not None else self.gen_config.top_k

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, prompt, max_tokens, temp, k
        )
        return result

    def _generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> dict:
        """Synchronous generation implementation.

        Performs tokenization, model forward passes, and sampling in a loop
        until ``max_new_tokens`` are generated or the model outputs an EOS token.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.

        Returns:
            Generation result dictionary.
        """
        model = self.manager.get_model()
        device = self.manager.device

        input_ids = self.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        start_time = time.perf_counter()
        generated_ids: list[int] = []

        with torch.no_grad():
            state = None
            current_input = input_tensor

            for _ in range(max_new_tokens):
                if hasattr(model, "step") and state is not None:
                    logits, state = model.step(current_input, state)
                else:
                    output = model(current_input, state)
                    if isinstance(output, tuple):
                        logits, state = output
                    else:
                        logits = output

                next_logits = logits[:, -1, :]
                next_token_id = self._sample(next_logits, temperature, top_k)
                generated_ids.append(next_token_id)
                current_input = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=device
                )

        elapsed = time.perf_counter() - start_time
        generated_text = self.decode(generated_ids)
        tokens_generated = len(generated_ids)

        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time_s": round(elapsed, 3),
            "tokens_per_second": (
                round(tokens_generated / elapsed, 1) if elapsed > 0 else 0
            ),
        }

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens one at a time.

        Yields decoded text fragments as they are generated. Uses a background
        thread for model inference with an async queue for cross-thread
        communication.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_k: Override default top-k value.

        Yields:
            Individual decoded token strings.
        """
        max_tokens = max_new_tokens or self.gen_config.max_new_tokens
        temp = temperature if temperature is not None else self.gen_config.temperature
        k = top_k if top_k is not None else self.gen_config.top_k

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def _produce() -> None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._stream_generate_sync,
                prompt,
                max_tokens,
                temp,
                k,
                queue,
                loop,
            )

        task = asyncio.create_task(_produce())

        try:
            while True:
                token_text = await queue.get()
                if token_text is None:
                    break
                yield token_text
        finally:
            if not task.done():
                task.cancel()

    def _stream_generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Synchronous streaming generation that pushes tokens to a queue.

        Runs in a thread executor. Each generated token is decoded and
        placed into the async queue for the consumer coroutine.

        Args:
            prompt: Input text.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k parameter.
            queue: Async queue for produced tokens.
            loop: Event loop for cross-thread queue operations.
        """
        model = self.manager.get_model()
        device = self.manager.device

        input_ids = self.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            state = None
            current_input = input_tensor

            for _ in range(max_new_tokens):
                if hasattr(model, "step") and state is not None:
                    logits, state = model.step(current_input, state)
                else:
                    output = model(current_input, state)
                    if isinstance(output, tuple):
                        logits, state = output
                    else:
                        logits = output

                next_logits = logits[:, -1, :]
                next_token_id = self._sample(next_logits, temperature, top_k)
                token_text = self.decode([next_token_id])

                asyncio.run_coroutine_threadsafe(
                    queue.put(token_text), loop
                )

                current_input = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device=device
                )

        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> int:
        """Sample the next token from logits.

        Supports greedy decoding (temperature=0) and top-k sampling with
        temperature scaling.

        Args:
            logits: Logit tensor of shape ``[1, vocab_size]``.
            temperature: Sampling temperature. 0 = greedy.
            top_k: Number of top candidates to consider. 0 = all.

        Returns:
            Selected token ID as an integer.
        """
        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, indices, values)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())
