"""Плановая очистка (pr-097): то, что копится без пользы и само не исчезает.

- Временные фото фоновой диагностики (`jobs/`): задача удаляет своё фото
  сама, но если воркер упал посреди анализа, файл остаётся навсегда.
- Истёкшие refresh-токены и ключи сброса пароля: войти по ним уже нельзя.
  Отозванные, но ещё не истёкшие токены не трогаются — по ним сервер
  узнаёт повторное использование украденного токена (pr-009).
- Прогноз погоды на прошедшие часы, который так и не заменил факт: архив
  Open-Meteo иногда отстаёт, и такой «прогноз задним числом» только путал
  бы выгрузки и графики.

Кэш в Redis и результаты задач Celery чистить не нужно: у них свой срок жизни."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, Delete, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.object_storage import Storage
from app.models.auth import PasswordResetToken, RefreshToken
from app.models.weather import WeatherData

JOBS_PREFIX = "jobs/"
# Фоновая диагностика занимает секунды: файл старше суток точно брошен
JOB_FILE_AGE = timedelta(days=1)
# Запас после истечения: часы сервера и базы могут немного расходиться
TOKEN_GRACE = timedelta(days=1)
# Архив догоняет прогноз за несколько дней; после недели его уже не будет
STALE_FORECAST = timedelta(days=7)


@dataclass(frozen=True)
class CleanupStats:
    job_files: int
    refresh_tokens: int
    reset_tokens: int
    forecast_hours: int


async def cleanup(
    session: AsyncSession, storage: Storage, now: datetime
) -> CleanupStats:
    job_files = await asyncio.to_thread(storage.purge, JOBS_PREFIX, now - JOB_FILE_AGE)
    refresh = await _delete(
        session, delete(RefreshToken).where(RefreshToken.expires_at < now - TOKEN_GRACE)
    )
    reset = await _delete(
        session,
        delete(PasswordResetToken).where(
            PasswordResetToken.expires_at < now - TOKEN_GRACE
        ),
    )
    forecast = await _delete(
        session,
        delete(WeatherData).where(
            WeatherData.is_forecast.is_(True),
            WeatherData.observed_at < now - STALE_FORECAST,
        ),
    )
    await session.commit()
    return CleanupStats(job_files, refresh, reset, forecast)


async def _delete(session: AsyncSession, statement: Delete) -> int:
    """Число удалённых строк."""
    result = cast(CursorResult[Any], await session.execute(statement))
    return result.rowcount
