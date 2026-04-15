# xLSTM Chat Documentation

Welcome to the **xLSTM Chat** documentation — a high-performance chat GUI and API
built on top of the [xLSTM](https://github.com/NX-AI/xlstm) architecture.

## Features

- **🖥️ Web Chat Interface** — Modern, responsive chat GUI with real-time streaming
- **⚡ High-Performance API** — FastAPI with WebSocket and SSE streaming
- **🔄 Model Switching** — Choose between small (NeurIPS) and large (7B) xLSTM models
- **🐳 Docker Ready** — Full Docker Compose deployment for CPU and GPU
- **📄 Auto-Documentation** — OpenAPI/Swagger docs generated from code
- **🔮 Future-Ready** — Extensibility hooks for Vision-LSTM support

## Quick Links

| Resource | URL |
|----------|-----|
| Chat GUI | `http://localhost:8000/` |
| Swagger Docs | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Health Check | `http://localhost:8000/api/v1/health` |

## Architecture

```
chat/
├── api/                    # FastAPI application and routes
│   ├── app.py              # Application factory
│   ├── dependencies.py     # Dependency injection
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── chat.py         # Chat endpoints (REST + WebSocket)
│       ├── models.py       # Model management endpoints
│       └── health.py       # Health and config endpoints
├── core/                   # Business logic
│   ├── config.py           # Configuration with env var support
│   ├── model_manager.py    # Model loading and lifecycle
│   └── inference.py        # Text generation engine
├── frontend/               # Web chat interface
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/chat.js
├── tests/                  # Test suite
├── Dockerfile              # Container build
├── docker-compose.yml      # Deployment orchestration
└── requirements.txt        # Python dependencies
```
