from collections.abc import Callable
from datetime import UTC, datetime

import httpx2 as httpx
import pytest

from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations.open_meteo import OpenMeteoClient

SETTINGS = Settings(_env_file=None)
CURRENT = {
    "latitude": 47.14,
    "longitude": 28.88,
    "current": {
        "time": "2026-10-09T11:30",
        "interval": 900,
        "temperature_2m": 24.2,
        "relative_humidity_2m": 43,
        "precipitation": 0.0,
    },
}

Handler = Callable[[httpx.Request], httpx.Response]


def client_with(handler: Handler) -> OpenMeteoClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenMeteoClient(http, SETTINGS)


async def test_current_weather_is_parsed_in_utc() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CURRENT)

    sample = await client_with(handler).current(47.14, 28.86)

    assert sample.observed_at == datetime(2026, 10, 9, 11, 30, tzinfo=UTC)
    assert sample.temperature_c == 24.2
    assert sample.relative_humidity == 43
    assert sample.precipitation_mm == 0
    params = requests[0].url.params
    assert str(requests[0].url).startswith("https://api.open-meteo.com/v1/forecast")
    assert params["latitude"] == "47.14"
    assert params["longitude"] == "28.86"
    assert params["timezone"] == "UTC"
    assert params["current"] == "temperature_2m,relative_humidity_2m,precipitation"


@pytest.mark.parametrize(
    "handler",
    [
        lambda _: httpx.Response(500, text="Internal Server Error"),
        lambda _: httpx.Response(200, text="<html>maintenance</html>"),
        lambda _: httpx.Response(200, json={"current": {"time": "2026-10-09T11:30"}}),
    ],
    ids=["server-error", "not-json", "missing-fields"],
)
async def test_bad_responses_become_external_service_error(handler: Handler) -> None:
    with pytest.raises(ExternalServiceError, match="Сервис погоды недоступен"):
        await client_with(handler).current(47.14, 28.86)


async def test_timeout_becomes_external_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ExternalServiceError):
        await client_with(handler).current(47.14, 28.86)
