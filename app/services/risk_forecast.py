"""Уточнение риска милдью моделью LightGBM на следующий день (pr-076).

Агромодели по прогнозу погоды остаются основой предупреждений: на 2-7 дней
модель по истории не лучше климатической частоты (ml/pr-018). На ближайший
день она точнее и показывается рядом как второе мнение."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.risk_model import RiskModel
from app.models.weather import WeatherData
from app.services.agromodels import (
    PRIMARY_INFECTION_END,
    HourlyWeather,
    in_window,
    label_days,
)
from app.services.risk_features import HISTORY_DAYS, vector


@dataclass(frozen=True)
class NextDayRisk:
    # День, на который сделан прогноз: следующий после последних полных суток
    day: date
    probability: float
    threshold: float
    high: bool
    model: str


async def next_day_infection(
    session: AsyncSession, vineyard_id: object, model: RiskModel, since: datetime
) -> NextDayRisk | None:
    """Прогноз по фактической погоде участка с начала сезона. Прогноза нет,
    если нет двух недель полных суток подряд или день прогноза вне окна
    «трёх десяток» (апрель - июнь): модель училась только на таких днях."""
    rows = await session.execute(
        select(
            WeatherData.observed_at,
            WeatherData.temperature_c,
            WeatherData.relative_humidity,
            WeatherData.precipitation_mm,
        )
        .where(
            WeatherData.vineyard_id == vineyard_id,
            WeatherData.is_forecast.is_(False),
            WeatherData.observed_at >= since,
        )
        .order_by(WeatherData.observed_at)
    )
    days = label_days(HourlyWeather(*row) for row in rows)
    if len(days) < HISTORY_DAYS:
        return None
    try:
        features = vector(days)
    except ValueError:
        return None
    last = date.fromisoformat(str(days[-1]["date"]))
    if not in_window(last + timedelta(days=1), PRIMARY_INFECTION_END):
        return None
    probability = model.probability(features)
    return NextDayRisk(
        day=last + timedelta(days=1),
        probability=probability,
        threshold=model.threshold,
        high=probability >= model.threshold,
        model=model.name,
    )
