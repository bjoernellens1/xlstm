# ── xLSTM Chat – Dockerfile ──────────────────────────────────────────────────
# Multi-stage build for production deployment.
# Supports both CPU and GPU (CUDA) via build args.
#
# Build:
#   docker build -t xlstm-chat .                       # CPU
#   docker build -t xlstm-chat --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 .  # GPU
#
# Run:
#   docker run -p 8000:8000 xlstm-chat
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim

# ── Stage 1: Dependencies ────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY chat/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Application ────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Install the xlstm package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Default environment variables
ENV XLSTM_CHAT_HOST=0.0.0.0 \
    XLSTM_CHAT_PORT=8000 \
    XLSTM_CHAT_LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "chat"]
