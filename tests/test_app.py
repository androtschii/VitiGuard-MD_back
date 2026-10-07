from fastapi.testclient import TestClient

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
