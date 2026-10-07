import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import __version__
from app.core.config import Settings
from app.main import create_app


def test_info(client: TestClient) -> None:
    response = client.get("/api/v1/info")

    assert response.status_code == 200
    assert response.json() == {
        "name": "VitiGuard MD API",
        "version": __version__,
        "environment": "test",
    }


def test_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["version"] == __version__
    assert "/api/v1/info" in schema["paths"]


def test_custom_api_prefix() -> None:
    app = create_app(Settings(_env_file=None, api_v1_prefix="/api/v2"))

    with TestClient(app) as client:
        assert client.get("/api/v2/info").status_code == 200
        assert client.get("/api/v1/info").status_code == 404


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITIGUARD_ENVIRONMENT", "staging")
    monkeypatch.setenv("VITIGUARD_DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.environment == "staging"
    assert settings.debug is True


def test_unknown_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITIGUARD_ENVIRONMENT", "prod")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
