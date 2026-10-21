"""Вероятность вспышки милдью и оидиума на 7 дней вперёд (pr-077).

Агромодели по одному прогнозу погоды дают ответ «да или нет», хотя дождь через
пять дней — уже не факт. Поэтому модели считаются по каждому члену ансамблевого
прогноза Open-Meteo, продолжающему фактическую погоду сезона, а вероятность
события в день — доля членов, где оно случилось. Вспышка — высокий уровень
риска: первичное заражение по «трём десяткам» или конец инкубации по модели
Гойданича для милдью, индекс Gubler-Thomas от 60 для оидиума."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.open_meteo import OpenMeteoClient, WeatherSample
from app.models.enums import Disease, RiskLevel
from app.models.vineyard import Vineyard
from app.models.weather import WeatherData
from app.services.agromodels import (
    LOCAL_TZ,
    SEASON_END,
    HourlyWeather,
    in_window,
    season_risks,
)
from app.services.risk import local_today, refresh_risks, season_year
from app.services.weather import vineyard_point

OUTLOOK_DAYS = 7
# Сохранённая погода может отставать на часы пересчёта (pr-079): ансамбль
# начинается раньше и закрывает разрыв до сегодняшнего дня
ENSEMBLE_PAST_DAYS = 2
# Седьмой местный день заканчивается уже в восьмых сутках по UTC
ENSEMBLE_FORECAST_DAYS = OUTLOOK_DAYS + 1


@dataclass(frozen=True)
class OutlookDay:
    day: date
    # None — день вне сезонного окна моделей болезни
    downy: float | None
    powdery: float | None


@dataclass(frozen=True)
class Outlook:
    members: int
    days: list[OutlookDay]
    # Вероятность хотя бы одной вспышки за неделю
    downy_week: float | None
    powdery_week: float | None


Events = dict[Disease, dict[date, bool]]


def member_events(hours: Sequence[HourlyWeather], days: Sequence[date]) -> Events:
    """Вспышки по дням в одном варианте погоды; дня нет — моделям нечего оценивать."""
    events: Events = {Disease.DOWNY_MILDEW: {}, Disease.POWDERY_MILDEW: {}}
    wanted = set(days)
    for risk in season_risks(hours):
        if risk.day in wanted:
            by_day = events[risk.disease]
            by_day[risk.day] = by_day.get(risk.day, False) or (
                risk.level is RiskLevel.HIGH
            )
    return events


def _share(flags: Sequence[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


def outlook_from_members(
    history: Sequence[HourlyWeather],
    members: Sequence[Sequence[HourlyWeather]],
    today: date,
) -> Outlook:
    """Каждый член ансамбля продолжает фактическую погоду с первого часа после
    неё; в общих часах верна история, а не прогноз."""
    days = [today + timedelta(days=offset) for offset in range(1, OUTLOOK_DAYS + 1)]
    last = history[-1].at if history else None
    runs = [
        member_events(
            [*history, *(h for h in member if last is None or h.at > last)], days
        )
        for member in members
    ]
    probabilities: dict[Disease, list[float | None]] = {}
    week: dict[Disease, float | None] = {}
    for disease in (Disease.DOWNY_MILDEW, Disease.POWDERY_MILDEW):
        probabilities[disease] = [
            _share([run[disease][day] for run in runs if day in run[disease]])
            for day in days
        ]
        week[disease] = _share(
            [any(run[disease].values()) for run in runs if run[disease]]
        )
    return Outlook(
        members=len(members),
        days=[
            OutlookDay(day, downy, powdery)
            for day, downy, powdery in zip(
                days,
                probabilities[Disease.DOWNY_MILDEW],
                probabilities[Disease.POWDERY_MILDEW],
                strict=True,
            )
        ],
        downy_week=week[Disease.DOWNY_MILDEW],
        powdery_week=week[Disease.POWDERY_MILDEW],
    )


def _hourly(sample: WeatherSample) -> HourlyWeather:
    return HourlyWeather(
        sample.observed_at,
        sample.temperature_c,
        sample.relative_humidity,
        sample.precipitation_mm,
    )


async def season_weather(
    session: AsyncSession, vineyard: Vineyard, today: date
) -> list[HourlyWeather]:
    """Фактическая погода сезона участка, сохранённая пересчётом риска."""
    start = datetime.combine(date(season_year(today), 3, 1), time(), LOCAL_TZ)
    rows = await session.execute(
        select(
            WeatherData.observed_at,
            WeatherData.temperature_c,
            WeatherData.relative_humidity,
            WeatherData.precipitation_mm,
        )
        .where(
            WeatherData.vineyard_id == vineyard.id,
            WeatherData.observed_at >= start.astimezone(UTC),
            WeatherData.is_forecast.is_(False),
        )
        .order_by(WeatherData.observed_at)
    )
    return [HourlyWeather(*row) for row in rows]


async def vineyard_outlook(
    session: AsyncSession,
    open_meteo: OpenMeteoClient,
    vineyard: Vineyard,
    now: datetime | None = None,
) -> Outlook:
    today = local_today(now)
    ahead = [today + timedelta(days=offset) for offset in range(1, OUTLOOK_DAYS + 1)]
    if not any(in_window(day, SEASON_END) for day in ahead):
        # Вне сезона прогнозировать нечего — ансамбль не запрашивается
        return outlook_from_members([], [], today)
    history = await season_weather(session, vineyard, today)
    if not history:
        # Без истории сезона модели не знают ни суммы температур (отросли ли
        # побеги), ни идущих инкубаций: сначала обычный пересчёт риска
        await refresh_risks(session, open_meteo, vineyard, now)
        history = await season_weather(session, vineyard, today)
    latitude, longitude = await vineyard_point(session, vineyard)
    members = await open_meteo.ensemble(
        latitude, longitude, ENSEMBLE_PAST_DAYS, ENSEMBLE_FORECAST_DAYS
    )
    return outlook_from_members(
        history, [[_hourly(sample) for sample in member] for member in members], today
    )
