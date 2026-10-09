from collections.abc import Iterator

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_open_meteo
from app.core.config import Settings
from app.integrations.open_meteo import OpenMeteoClient

pytestmark = pytest.mark.integration

URL = "/api/v1/weather/current"
CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}
CURRENT = {
    "current": {
        "time": "2026-10-09T11:30",
        "temperature_2m": 24.2,
        "relative_humidity_2m": 43,
        "precipitation": 0.0,
    }
}


@pytest.fixture
def weather_status() -> list[int]:
    """Код ответа подменного Open-Meteo; тест может его поменять."""
    return [200]


@pytest.fixture
def client(api_client: TestClient, weather_status: list[int]) -> Iterator[TestClient]:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(weather_status[0], json=CURRENT)

    mock = OpenMeteoClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        Settings(_env_file=None),
    )
    app = api_client.app
    app.dependency_overrides[get_open_meteo] = lambda: mock  # type: ignore[attr-defined]
    yield api_client
    app.dependency_overrides.clear()  # type: ignore[attr-defined]


@pytest.fixture
def headers(client: TestClient) -> dict[str, str]:
    client.post("/api/v1/auth/register", json={**CREDENTIALS, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=CREDENTIALS).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_current_weather(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get(
        URL, params={"latitude": 47.14, "longitude": 28.86}, headers=headers
    )

    assert response.status_code == 200
    assert response.json() == {
        "observed_at": "2026-10-09T11:30:00Z",
        "temperature_c": 24.2,
        "relative_humidity": 43.0,
        "precipitation_mm": 0.0,
    }


def test_weather_requires_login(client: TestClient) -> None:
    response = client.get(URL, params={"latitude": 47.14, "longitude": 28.86})

    assert response.status_code == 401


@pytest.mark.parametrize(
    "params",
    [{"latitude": 91, "longitude": 28}, {"latitude": 47, "longitude": 181}, {}],
    ids=["latitude", "longitude", "missing"],
)
def test_invalid_coordinates_give_422(
    client: TestClient, headers: dict[str, str], params: dict[str, float]
) -> None:
    assert client.get(URL, params=params, headers=headers).status_code == 422


def test_weather_service_failure_gives_502(
    client: TestClient, headers: dict[str, str], weather_status: list[int]
) -> None:
    weather_status[0] = 503

    response = client.get(
        URL, params={"latitude": 47.14, "longitude": 28.86}, headers=headers
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Сервис погоды недоступен, попробуйте позже"}
