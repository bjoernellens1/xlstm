# Chat Endpoints

## POST /api/v1/chat

Generate a chat completion response.

### Request Body

```json
{
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_k": 50,
    "stream": false
}
```

### Response (non-streaming)

```json
{
    "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!"
    },
    "usage": {
        "tokens_generated": 42,
        "generation_time_s": 1.234,
        "tokens_per_second": 34.0
    }
}
```

### Response (streaming, `stream: true`)

Returns `text/event-stream` with Server-Sent Events:

```
data: {"token": "I"}

data: {"token": "'m"}

data: {"token": " doing"}

data: [DONE]
```

## WebSocket /api/v1/chat/ws

Real-time streaming chat via WebSocket for lower latency.

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/chat/ws");
```

### Send Message

```json
{
    "messages": [
        {"role": "user", "content": "Tell me a story"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_k": 50
}
```

### Receive Tokens

Each generated token arrives as a JSON message:

```json
{"token": "Once"}
{"token": " upon"}
{"token": " a"}
{"token": " time"}
```

Generation complete signal:

```json
{"done": true}
```

### Error Handling

```json
{"error": "Model not loaded. Load a model first."}
```

::: chat.api.routes.chat
