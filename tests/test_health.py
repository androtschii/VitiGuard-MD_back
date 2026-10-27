from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

# Порт 1 закрыт: соединение отклоняется сразу, без ожидания тайм-аута
DOWN_DATABASE = "postgresql+asyncpg://vitiguard:vitiguard@127.0.0.1:1/vitiguard"
DOWN_REDIS = "redis://127.0.0.1:1/2"


def make_client(database_url: str, cache_url: str) -> TestClient:
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=database_url,  # type: ignore[arg-type]
        cache_url=cache_url,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings))


@pytest.fixture
def down_client() -> Iterator[TestClient]:
    with make_client(DOWN_DATABASE, DOWN_REDIS) as client:
        yield client


def test_healthz_does_not_touch_database_or_redis(down_client: TestClient) -> None:
    response = down_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_unavailable_services(down_client: TestClient) -> None:
    response = down_client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "error", "redis": "error"},
    }


def test_health_routes_are_outside_api_prefix(down_client: TestClient) -> None:
    assert down_client.get("/api/v1/healthz").status_code == 404
    paths = down_client.get("/openapi.json").json()["paths"]
    assert {"/healthz", "/readyz"} <= set(paths)


@pytest.mark.integration
def test_readyz_when_database_and_redis_are_up(
    database_url: str, redis_url: str
) -> None:
    with make_client(database_url, f"{redis_url}/2") as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"database": "ok", "redis": "ok"},
    }


@pytest.mark.integration
def test_readyz_names_the_failed_service(database_url: str) -> None:
    with make_client(database_url, DOWN_REDIS) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"] == {"database": "ok", "redis": "error"}
