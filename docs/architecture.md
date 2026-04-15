# Architecture

## Overview

xLSTM Chat is designed with a clean separation of concerns for maintainability
and future extensibility.

```
┌─────────────────────────────────────────────────────┐
│                    Web Frontend                      │
│             (HTML/CSS/JavaScript)                     │
│    ┌──────────┐  ┌──────────────┐  ┌──────────┐     │
│    │ Chat UI  │  │   Settings   │  │ Model    │     │
│    │          │  │   Panel      │  │ Selector │     │
│    └────┬─────┘  └──────┬───────┘  └────┬─────┘     │
│         │               │               │            │
│         └───────────────┼───────────────┘            │
│                   WebSocket / REST                    │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────┐
│                   FastAPI Server                     │
│  ┌─────────────┐ ┌────────────┐ ┌───────────────┐   │
│  │ Chat Routes │ │Model Routes│ │ Health Routes │   │
│  └──────┬──────┘ └─────┬──────┘ └───────┬───────┘   │
│         │              │                │            │
│  ┌──────┴──────────────┴────────────────┴───────┐   │
│  │              Dependencies Layer               │   │
│  │    (Dependency Injection / Shared State)       │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────┼───────────────────────┐   │
│  │                 Core Layer                    │   │
│  │  ┌──────────────┐  ┌────────────────────┐    │   │
│  │  │Model Manager │  │ Inference Engine   │    │   │
│  │  │              │  │                    │    │   │
│  │  │ • load/unload│  │ • generate()       │    │   │
│  │  │ • switch     │  │ • generate_stream()│    │   │
│  │  │ • device mgmt│  │ • sampling         │    │   │
│  │  └──────┬───────┘  └────────┬───────────┘    │   │
│  │         │                   │                 │   │
│  │  ┌──────┴───────────────────┴───────────┐    │   │
│  │  │            Configuration             │    │   │
│  │  │  (env vars, CLI args, defaults)      │    │   │
│  │  └──────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────┐
│                  xLSTM Models                        │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────┐  │
│  │  xLSTM Small │  │  xLSTM Large  │  │ Vision   │  │
│  │  (NeurIPS)   │  │  (7B)         │  │ (future) │  │
│  └──────────────┘  └───────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. FastAPI for the API Layer

- **Automatic OpenAPI documentation** — Swagger UI and ReDoc generated from code
- **Async support** — Non-blocking I/O for WebSocket and SSE
- **Type safety** — Pydantic schemas for request/response validation
- **High performance** — One of the fastest Python web frameworks

### 2. Vanilla Frontend (No Framework)

- **Zero build step** — No npm, webpack, or bundler needed
- **Minimal dependencies** — Pure HTML/CSS/JS
- **Easy to customize** — Standard web technologies
- **Fast loading** — No framework overhead

### 3. WebSocket for Streaming

- **Lower latency** — Persistent connection, no HTTP overhead per token
- **Bidirectional** — Can send/receive without polling
- **SSE fallback** — REST endpoint supports Server-Sent Events as fallback

### 4. Singleton Model Manager

- **Thread-safe** — Uses asyncio.Lock for concurrent access
- **Memory efficient** — Only one model loaded at a time
- **Clean lifecycle** — Explicit load/unload with GPU memory cleanup

## Adding New Model Variants

To add a new model variant (e.g., Vision-LSTM):

1. **Add variant to enums** in `chat/core/config.py`:
   ```python
   class ModelVariant(str, Enum):
       SMALL = "small"
       LARGE = "large"
       VISION = "vision"  # New variant
   ```

2. **Add loading logic** in `chat/core/model_manager.py`:
   ```python
   async def load_model(self, variant):
       ...
       elif target_variant == ModelVariant.VISION:
           await loop.run_in_executor(None, self._load_vision_model)
   ```

3. **Update the variants list** in `ModelManager.available_variants()`

4. **Add API schema option** in `chat/api/schemas.py`

## Configuration Flow

```
Environment Variables → load_config() → AppConfig
                                           ├── ModelConfig
CLI Arguments ─────────────────────────────├── GenerationConfig
                                           └── ServerConfig
```

Settings are resolved in priority order:
1. CLI arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)
