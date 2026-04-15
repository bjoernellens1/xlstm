"""Tests for the xLSTM Chat API.

Tests cover all REST endpoints including health checks, model management,
generation configuration, and chat completions. Uses httpx async test client
with FastAPI's TestClient approach.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from chat.api.app import create_app
from chat.core.config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    """Create a test configuration."""
    return AppConfig()


@pytest.fixture
def client(app_config: AppConfig) -> TestClient:
    """Create a test client with the app, triggering lifespan events."""
    app = create_app(app_config)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint should return 200 with status info."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert "version" in data

    def test_health_check_no_model(self, client: TestClient) -> None:
        """Health should report model_loaded=false when no model is loaded."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["model_loaded"] is False


class TestModelEndpoints:
    """Tests for the /api/v1/models endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        """Should return a list of available model variants."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        names = [m["name"] for m in data]
        assert "small" in names
        assert "large" in names

    def test_list_models_includes_vision(self, client: TestClient) -> None:
        """Should include planned vision variant."""
        response = client.get("/api/v1/models")
        data = response.json()
        vision = [m for m in data if m["name"] == "vision"]
        assert len(vision) == 1
        assert vision[0]["status"] == "planned"

    def test_get_current_model_none(self, client: TestClient) -> None:
        """Should report no model loaded initially."""
        response = client.get("/api/v1/models/current")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is False

    def test_unload_model(self, client: TestClient) -> None:
        """Unload should succeed even when no model is loaded."""
        response = client.post("/api/v1/models/unload")
        assert response.status_code == 200


class TestConfigEndpoints:
    """Tests for the /api/v1/config endpoints."""

    def test_get_generation_config(self, client: TestClient) -> None:
        """Should return current generation config."""
        response = client.get("/api/v1/config/generation")
        assert response.status_code == 200
        data = response.json()
        assert "max_new_tokens" in data
        assert "temperature" in data
        assert "top_k" in data
        assert "top_p" in data
        assert "repetition_penalty" in data

    def test_update_generation_config(self, client: TestClient) -> None:
        """Should update only the provided fields."""
        response = client.patch(
            "/api/v1/config/generation",
            json={"temperature": 0.5, "max_new_tokens": 128},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["temperature"] == 0.5
        assert data["max_new_tokens"] == 128

    def test_update_generation_config_partial(self, client: TestClient) -> None:
        """Should only update specified fields, leaving others unchanged."""
        # Get original config
        original = client.get("/api/v1/config/generation").json()

        # Update only temperature
        response = client.patch(
            "/api/v1/config/generation",
            json={"temperature": 1.5},
        )
        data = response.json()
        assert data["temperature"] == 1.5
        assert data["top_k"] == original["top_k"]


class TestChatEndpoint:
    """Tests for the /api/v1/chat endpoint."""

    def test_chat_without_model(self, client: TestClient) -> None:
        """Should return 503 when no model is loaded."""
        response = client.post(
            "/api/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 503

    def test_chat_empty_messages(self, client: TestClient) -> None:
        """Should return 422 for empty messages list."""
        response = client.post(
            "/api/v1/chat",
            json={"messages": []},
        )
        assert response.status_code == 422


class TestFrontend:
    """Tests for the frontend serving."""

    def test_serve_index(self, client: TestClient) -> None:
        """Should serve the chat HTML page at root."""
        response = client.get("/")
        assert response.status_code == 200
        assert "xLSTM Chat" in response.text

    def test_serve_css(self, client: TestClient) -> None:
        """Should serve static CSS file."""
        response = client.get("/static/css/style.css")
        assert response.status_code == 200

    def test_serve_js(self, client: TestClient) -> None:
        """Should serve static JS file."""
        response = client.get("/static/js/chat.js")
        assert response.status_code == 200
