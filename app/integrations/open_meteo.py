"""Клиент Open-Meteo: текущая погода и почасовой архив по координатам.

Сервис бесплатный и без ключа. Все запросы — в UTC, чтобы время в базе не
зависело от часового пояса и перехода на летнее время."""

from datetime import UTC, datetime
from typing import Any

import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import ExternalServiceError

VARIABLES = "temperature_2m,relative_humidity_2m,precipitation"
UNAVAILABLE = "Сервис погоды недоступен, попробуйте позже"


class WeatherSample(BaseModel):
    """Погода в точке в момент времени."""

    observed_at: datetime
    temperature_c: float
    relative_humidity: float
    precipitation_mm: float


class _Current(BaseModel):
    time: datetime
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float


class _CurrentResponse(BaseModel):
    current: _Current


class OpenMeteoClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    async def current(self, latitude: float, longitude: float) -> WeatherSample:
        data = await self._get(
            self._settings.open_meteo_forecast_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": VARIABLES,
                "timezone": "UTC",
            },
        )
        try:
            current = _CurrentResponse.model_validate(data).current
        except ValidationError as error:
            raise ExternalServiceError(UNAVAILABLE) from error
        return WeatherSample(
            observed_at=_as_utc(current.time),
            temperature_c=current.temperature_2m,
            relative_humidity=current.relative_humidity_2m,
            precipitation_mm=current.precipitation,
        )

    async def _get(self, url: str, params: dict[str, Any]) -> Any:
        """Запрос к Open-Meteo. Сетевой сбой, таймаут, ошибка сервиса или
        ответ не в JSON — ExternalServiceError (ответ API 502)."""
        try:
            response = await self._http.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ExternalServiceError(UNAVAILABLE) from error


def _as_utc(moment: datetime) -> datetime:
    # С timezone=UTC сервис отдаёт время без пояса; хранить его надо явно в UTC
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment
