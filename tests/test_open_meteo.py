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


ENSEMBLE = {
    "hourly": {
        "time": ["2026-06-10T00:00", "2026-06-10T01:00"],
        "temperature_2m": [18.0, 17.5],
        "relative_humidity_2m": [80, 82],
        "precipitation": [0.0, 0.2],
        "temperature_2m_member01": [18.4, 17.9],
        "relative_humidity_2m_member01": [78, 79],
        "precipitation_member01": [0.0, None],
        "temperature_2m_member02": [17.6, 17.0],
        "relative_humidity_2m_member02": [85, 88],
        "precipitation_member02": [1.2, 0.8],
    }
}


async def test_ensemble_gives_series_per_member() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=ENSEMBLE)

    members = await client_with(handler).ensemble(47.14, 28.86, 2, 8)

    # Контрольный прогноз и два возмущённых; час без значения пропускается
    assert [len(member) for member in members] == [2, 1, 2]
    assert [member[0].temperature_c for member in members] == [18.0, 18.4, 17.6]
    assert members[2][1].precipitation_mm == 0.8
    assert members[0][0].observed_at == datetime(2026, 6, 10, tzinfo=UTC)
    params = requests[0].url.params
    assert str(requests[0].url).startswith(
        "https://ensemble-api.open-meteo.com/v1/ensemble"
    )
    assert params["models"] == "icon_seamless"
    assert params["past_days"] == "2"
    assert params["forecast_days"] == "8"
    assert params["hourly"] == "temperature_2m,relative_humidity_2m,precipitation"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"hourly": []},
        {"hourly": {"time": ["2026-06-10T00:00"]}},
        {
            "hourly": {
                "time": ["2026-06-10T00:00"],
                "temperature_2m_member01": [18.0],
                "precipitation_member01": [0.0],
            }
        },
    ],
    ids=["not-object", "no-hourly", "no-members", "member-without-humidity"],
)
async def test_bad_ensemble_becomes_external_service_error(payload: object) -> None:
    with pytest.raises(ExternalServiceError):
        await client_with(lambda _: httpx.Response(200, json=payload)).ensemble(
            47.14, 28.86, 2, 8
        )
