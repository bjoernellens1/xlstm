# API Overview

xLSTM Chat provides a complete REST and WebSocket API with automatic documentation.

## Interactive Documentation

When the server is running, access the interactive API docs at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These are auto-generated from the code and always up-to-date.

## API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Generate a chat response |
| `WS` | `/api/v1/chat/ws` | WebSocket for streaming chat |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/models` | List available model variants |
| `GET` | `/api/v1/models/current` | Get current model info |
| `POST` | `/api/v1/models/load` | Load a model variant |
| `POST` | `/api/v1/models/unload` | Unload the current model |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config/generation` | Get generation config |
| `PATCH` | `/api/v1/config/generation` | Update generation config |

## Authentication

Currently, no authentication is required. For production deployments,
consider adding API key authentication or OAuth2 via a reverse proxy.

## Error Handling

All errors return JSON responses:

```json
{
    "detail": "Error description"
}
```

Standard HTTP status codes are used:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request / Invalid parameters |
| 422 | Validation error |
| 503 | Service unavailable (model not loaded) |
