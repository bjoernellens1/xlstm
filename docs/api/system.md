# System Endpoints

## GET /api/v1/health

Health check endpoint for monitoring and load balancers.

### Response

```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "0.1.0"
}
```

## GET /api/v1/config/generation

Get current text generation configuration.

### Response

```json
{
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

## PATCH /api/v1/config/generation

Update generation configuration at runtime. Only fields included in the
request body are modified.

### Request Body

```json
{
    "temperature": 0.5,
    "max_new_tokens": 128
}
```

### Response

Returns the full updated configuration.

::: chat.api.routes.health
