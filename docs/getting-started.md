# Getting Started

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- (Optional) NVIDIA GPU with CUDA for the large model

## Installation

### From the repository

```bash
# Clone the repository
git clone https://github.com/bjoernellens1/xlstm.git
cd xlstm

# Install the base xlstm package
pip install -e .

# Install chat dependencies
pip install -r chat/requirements.txt
```

### Using Docker

```bash
cd chat
docker compose up --build
```

## Running the Chat Server

### Command Line

```bash
# Start with default settings (small model, CPU)
python -m chat

# Start with custom settings
python -m chat --port 8080 --variant small --device cpu

# Start with large model
python -m chat --variant large --checkpoint /path/to/checkpoint
```

### Environment Variables

All settings can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `XLSTM_CHAT_HOST` | `0.0.0.0` | Server bind address |
| `XLSTM_CHAT_PORT` | `8000` | Server port |
| `XLSTM_CHAT_MODEL_VARIANT` | `small` | Model variant (`small` or `large`) |
| `XLSTM_CHAT_DEVICE` | `auto` | Compute device (`cpu`, `cuda`, `auto`) |
| `XLSTM_CHAT_CHECKPOINT_PATH` | - | Path to model checkpoint (large variant) |
| `XLSTM_CHAT_MAX_NEW_TOKENS` | `256` | Default max tokens per response |
| `XLSTM_CHAT_TEMPERATURE` | `0.7` | Default sampling temperature |
| `XLSTM_CHAT_TOP_K` | `50` | Default top-k sampling |
| `XLSTM_CHAT_LOG_LEVEL` | `info` | Logging level |

## Using the Chat Interface

1. Open `http://localhost:8000` in your browser
2. Select a model variant from the sidebar (small or large)
3. Click **Load Model** to initialize
4. Start chatting!

### Generation Settings

Adjust these in the sidebar:

- **Temperature**: Controls randomness (0 = deterministic, 2 = very random)
- **Max Tokens**: Maximum response length
- **Top-K**: Limits sampling to top K most likely tokens

## Using the API

### Chat Completion

```bash
# Non-streaming
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_new_tokens": 100,
    "temperature": 0.7
  }'

# Streaming (SSE)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Model Management

```bash
# List available models
curl http://localhost:8000/api/v1/models

# Load a model
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"variant": "small"}'

# Get current model info
curl http://localhost:8000/api/v1/models/current
```

See the [API Reference](api/overview.md) for complete documentation.
