# Model Endpoints

## GET /api/v1/models

List all available model variants.

### Response

```json
[
    {
        "name": "small",
        "description": "xLSTM NeurIPS model - lightweight, CPU-friendly",
        "status": "available"
    },
    {
        "name": "large",
        "description": "xLSTM Large 7B - high-quality, GPU recommended",
        "status": "available"
    },
    {
        "name": "vision",
        "description": "Vision-LSTM - xLSTM as Generic Vision Backbone",
        "status": "planned"
    }
]
```

## GET /api/v1/models/current

Get information about the currently loaded model.

### Response (model loaded)

```json
{
    "loaded": true,
    "variant": "small",
    "device": "cpu",
    "parameter_count": 8523776,
    "parameter_count_human": "8.5M",
    "vocab_size": 50304,
    "max_context_length": 256,
    "has_tokenizer": true,
    "supports_vision": false
}
```

### Response (no model loaded)

```json
{
    "loaded": false,
    "variant": null
}
```

## POST /api/v1/models/load

Load or switch to a specific model variant.

### Request Body

```json
{
    "variant": "small"
}
```

### Response

Same as `GET /api/v1/models/current` — returns info about the newly loaded model.

## POST /api/v1/models/unload

Unload the current model and free memory.

### Response

```json
{
    "message": "Model unloaded successfully."
}
```

::: chat.api.routes.models
