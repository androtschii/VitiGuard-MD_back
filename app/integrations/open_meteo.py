"""Клиент Open-Meteo: текущая погода, почасовой архив, прогноз и ансамблевый
прогноз по координатам.

Сервис бесплатный и без ключа. Все запросы — в UTC, чтобы время в базе не
зависело от часового пояса и перехода на летнее время."""

from datetime import UTC, date, datetime
from typing import Any, Self

import httpx2 as httpx
from pydantic import BaseModel, ValidationError, model_validator

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


class _Hourly(BaseModel):
    time: list[datetime]
    temperature_2m: list[float | None]
    relative_humidity_2m: list[float | None]
    precipitation: list[float | None]

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.relative_humidity_2m),
            len(self.precipitation),
        }
        if len(lengths) != 1:
            raise ValueError("Ряды почасовых данных разной длины")
        return self


class _HourlyResponse(BaseModel):
    hourly: _Hourly


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

    async def history(
        self, latitude: float, longitude: float, start: date, end: date
    ) -> list[WeatherSample]:
        """Почасовой архив за дни с start по end включительно (UTC).

        Часы, для которых в архиве ещё нет значений (последние дни приходят с
        задержкой), пропускаются: ноль вместо пропуска исказил бы агромодели —
        «0 мм осадков» и «нет данных» — не одно и то же."""
        data = await self._get(
            self._settings.open_meteo_archive_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": VARIABLES,
                "timezone": "UTC",
            },
        )
        return _hourly_samples(data)

    async def forecast(
        self, latitude: float, longitude: float, past_days: int, forecast_days: int
    ) -> list[WeatherSample]:
        """Почасовая погода за последние past_days дней и прогноз на
        forecast_days вперёд одним запросом. Архив приходит с задержкой в
        несколько дней — последние дни берутся отсюда."""
        data = await self._get(
            self._settings.open_meteo_forecast_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "past_days": past_days,
                "forecast_days": forecast_days,
                "hourly": VARIABLES,
                "timezone": "UTC",
            },
        )
        return _hourly_samples(data)

    async def ensemble(
        self, latitude: float, longitude: float, past_days: int, forecast_days: int
    ) -> list[list[WeatherSample]]:
        """Ансамблевый прогноз: почасовой ряд на каждого члена ансамбля —
        контрольный прогноз и прогнозы с возмущёнными начальными условиями.
        Разброс членов показывает, насколько неопределённа погода впереди."""
        data = await self._get(
            self._settings.open_meteo_ensemble_url,
            {
                "latitude": latitude,
                "longitude": longitude,
                "models": self._settings.open_meteo_ensemble_model,
                "past_days": past_days,
                "forecast_days": forecast_days,
                "hourly": VARIABLES,
                "timezone": "UTC",
            },
        )
        return _ensemble_members(data)

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


def _ensemble_members(data: Any) -> list[list[WeatherSample]]:
    """Члены ансамбля приходят столбцами одного ответа: temperature_2m —
    контрольный прогноз, temperature_2m_member01 и дальше — остальные."""
    hourly = data.get("hourly") if isinstance(data, dict) else None
    if not isinstance(hourly, dict):
        raise ExternalServiceError(UNAVAILABLE)
    suffixes = sorted(
        key.removeprefix("temperature_2m")
        for key in hourly
        if key.startswith("temperature_2m")
    )
    if not suffixes:
        raise ExternalServiceError(UNAVAILABLE)
    return [
        _hourly_samples(
            {
                "hourly": {
                    "time": hourly.get("time"),
                    "temperature_2m": hourly.get(f"temperature_2m{suffix}"),
                    "relative_humidity_2m": hourly.get(f"relative_humidity_2m{suffix}"),
                    "precipitation": hourly.get(f"precipitation{suffix}"),
                }
            }
        )
        for suffix in suffixes
    ]


def _hourly_samples(data: Any) -> list[WeatherSample]:
    """Часы, для которых нет значений, пропускаются: ноль вместо пропуска
    исказил бы агромодели — «0 мм осадков» и «нет данных» — не одно и то же."""
    try:
        hourly = _HourlyResponse.model_validate(data).hourly
    except ValidationError as error:
        raise ExternalServiceError(UNAVAILABLE) from error
    samples = []
    for moment, temperature, humidity, precipitation in zip(
        hourly.time,
        hourly.temperature_2m,
        hourly.relative_humidity_2m,
        hourly.precipitation,
        strict=True,
    ):
        if temperature is None or humidity is None or precipitation is None:
            continue
        samples.append(
            WeatherSample(
                observed_at=_as_utc(moment),
                temperature_c=temperature,
                relative_humidity=humidity,
                precipitation_mm=precipitation,
            )
        )
    return samples
