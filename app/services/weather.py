from datetime import date, timedelta

from geoalchemy2 import functions as geo
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import BadRequestError, NotFoundError
from app.integrations.open_meteo import OpenMeteoClient
from app.models.vineyard import Vineyard
from app.repositories.weather import WeatherRepository

# Не больше года за раз: ответ архива остаётся небольшим, а длинную историю
# фоновая задача загрузит по частям
MAX_HISTORY_DAYS = 366


async def vineyard_point(
    session: AsyncSession, vineyard: Vineyard
) -> tuple[float, float]:
    """Точка участка для запроса погоды: (широта, долгота) внутри полигона.
    ST_PointOnSurface, а не центроид: у участка сложной формы (например, в виде
    буквы «С») центроид может оказаться за его границей."""
    point = geo.ST_PointOnSurface(Vineyard.geom)
    row = (
        await session.execute(
            select(geo.ST_Y(point), geo.ST_X(point)).where(Vineyard.id == vineyard.id)
        )
    ).one_or_none()
    if row is None:
        raise NotFoundError("Участок не найден")
    latitude, longitude = row
    return float(latitude), float(longitude)


async def import_weather_history(
    session: AsyncSession,
    open_meteo: OpenMeteoClient,
    vineyard: Vineyard,
    start: date,
    end: date,
) -> int:
    """Загружает почасовую погоду участка за период и возвращает число часов."""
    if start > end:
        raise BadRequestError("Начало периода позже конца")
    if end - start >= timedelta(days=MAX_HISTORY_DAYS):
        raise BadRequestError(f"Период не длиннее {MAX_HISTORY_DAYS} дней")
    latitude, longitude = await vineyard_point(session, vineyard)
    samples = await open_meteo.history(latitude, longitude, start, end)
    count = await WeatherRepository(session).upsert(vineyard.id, samples)
    await session.commit()
    return count
