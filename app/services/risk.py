"""Риск милдью и оидиума на участке по агромоделям (pr-073, pr-074, pr-077).

Погода сезона: архив Open-Meteo с 1 марта (сумма активных температур считается
с марта) и последние дни с прогнозом на неделю вперёд — архив приходит с
задержкой в несколько дней. По этому ряду считаются модели, и риск сохраняется
по дням: прошедшие дни — факт, следующие семь — прогноз по прогнозу погоды."""

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.open_meteo import OpenMeteoClient, WeatherSample
from app.models.vineyard import Vineyard
from app.models.weather import DiseaseRisk
from app.repositories.weather import WeatherRepository
from app.services.agromodels import (
    LOCAL_TZ,
    SEASON_START,
    HourlyWeather,
    RiskDay,
    season_risks,
)
from app.services.weather import vineyard_point

FORECAST_DAYS = 7
# Последняя неделя — из сервиса прогноза: в архиве её ещё может не быть
RECENT_DAYS = 7


def local_today(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(LOCAL_TZ).date()


def season_year(today: date) -> int:
    # До марта текущий сезон ещё не начался — показывается прошлый
    return today.year if today.month >= 3 else today.year - 1


def merge_weather(
    archive: Sequence[WeatherSample], recent: Sequence[WeatherSample]
) -> list[WeatherSample]:
    """Один ряд без повторов часов. Где есть оба источника, берётся архив: это
    реанализ по наблюдениям, а последние дни сервиса прогноза — модельная оценка."""
    hours = {sample.observed_at: sample for sample in recent}
    hours.update({sample.observed_at: sample for sample in archive})
    return [hours[moment] for moment in sorted(hours)]


async def refresh_risks(
    session: AsyncSession,
    open_meteo: OpenMeteoClient,
    vineyard: Vineyard,
    now: datetime | None = None,
) -> int:
    """Загружает погоду сезона, пересчитывает и сохраняет риск по дням.
    Возвращает число сохранённых оценок."""
    moment = now or datetime.now(UTC)
    today = local_today(moment)
    year = season_year(today)
    latitude, longitude = await vineyard_point(session, vineyard)
    archive_end = min(today - timedelta(days=RECENT_DAYS), date(year, 12, 31))
    archive = (
        await open_meteo.history(latitude, longitude, date(year, 3, 1), archive_end)
        if archive_end >= date(year, 3, 1)
        else []
    )
    recent = (
        await open_meteo.forecast(latitude, longitude, RECENT_DAYS, FORECAST_DAYS)
        if year == today.year
        else []
    )
    weather = merge_weather(archive, recent)

    repository = WeatherRepository(session)
    await repository.upsert(
        vineyard.id, [s for s in weather if s.observed_at <= moment]
    )
    await repository.upsert(
        vineyard.id,
        [s for s in weather if s.observed_at > moment],
        is_forecast=True,
    )
    risks = season_risks(
        HourlyWeather(
            s.observed_at, s.temperature_c, s.relative_humidity, s.precipitation_mm
        )
        for s in weather
    )
    await _save(session, vineyard, risks, today)
    await session.commit()
    return len(risks)


async def _save(
    session: AsyncSession, vineyard: Vineyard, risks: Sequence[RiskDay], today: date
) -> None:
    if not risks:
        return
    rows = [
        {
            "vineyard_id": vineyard.id,
            "disease": risk.disease,
            "target_date": risk.day,
            "method": risk.method,
            "score": risk.score,
            "level": risk.level,
            # Дни после сегодняшнего посчитаны по прогнозу погоды
            "factors": {**risk.factors, "forecast": risk.day > today},
        }
        for risk in risks
    ]
    # Пачками: сезон трёх моделей — около 500 строк
    for start in range(0, len(rows), 200):
        statement = insert(DiseaseRisk).values(rows[start : start + 200])
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=["vineyard_id", "disease", "target_date", "method"],
                set_={
                    "score": statement.excluded.score,
                    "level": statement.excluded.level,
                    "factors": statement.excluded.factors,
                    "updated_at": func.now(),
                },
            )
        )


async def season_risk_history(
    session: AsyncSession, vineyard: Vineyard, today: date | None = None
) -> Sequence[DiseaseRisk]:
    """Сохранённый риск текущего сезона по дате и методу."""
    year = season_year(today or local_today())
    rows = await session.scalars(
        select(DiseaseRisk)
        .where(
            DiseaseRisk.vineyard_id == vineyard.id,
            DiseaseRisk.target_date >= date(year, *SEASON_START),
            DiseaseRisk.target_date <= date(year, 12, 31),
        )
        .order_by(DiseaseRisk.target_date, DiseaseRisk.method)
    )
    return rows.all()
