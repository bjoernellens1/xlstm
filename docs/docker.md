# Docker Deployment

xLSTM Chat provides full Docker support for both CPU and GPU deployments.

## Quick Start

```bash
cd chat

# CPU mode (default)
docker compose up --build

# GPU mode (requires NVIDIA Docker runtime)
docker compose --profile gpu up --build
```

## Build Options

### CPU Image

```bash
docker build -t xlstm-chat -f chat/Dockerfile .
```

### GPU Image

```bash
docker build -t xlstm-chat-gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 \
  -f chat/Dockerfile .
```

## Docker Compose Configuration

The `docker-compose.yml` defines two services:

| Service | Profile | Device | Description |
|---------|---------|--------|-------------|
| `xlstm-chat` | default | CPU | Lightweight, always available |
| `xlstm-chat-gpu` | `gpu` | CUDA | Requires NVIDIA GPU + Docker runtime |

### Environment Configuration

Create a `.env` file in the `chat/` directory:

```env
XLSTM_CHAT_PORT=8000
XLSTM_CHAT_MODEL_VARIANT=small
XLSTM_CHAT_MAX_NEW_TOKENS=256
XLSTM_CHAT_TEMPERATURE=0.7
XLSTM_CHAT_LOG_LEVEL=info
```

For GPU deployments with a pretrained checkpoint:

```env
XLSTM_CHAT_MODEL_VARIANT=large
XLSTM_CHAT_CHECKPOINT_PATH=/app/models/xlstm-7b
```

### Volumes

The `model-cache` volume persists downloaded model files across container restarts:

```yaml
volumes:
  model-cache:
    driver: local
```

## Health Checks

Both services include health checks that query the `/api/v1/health` endpoint:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' xlstm-chat
```

## Production Deployment

For production, consider:

1. **Reverse Proxy**: Use nginx or traefik in front of the service
2. **TLS**: Enable HTTPS via the reverse proxy
3. **Logging**: Mount a log directory or use a logging driver
4. **Monitoring**: Add Prometheus metrics export

Example nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name chat.example.com;

    location / {
        proxy_pass http://xlstm-chat:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```
