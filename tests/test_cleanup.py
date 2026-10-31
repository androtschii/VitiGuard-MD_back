import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from geoalchemy2 import WKTElement
from memory_storage import MemoryStorage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.integrations.object_storage import ObjectStorage
from app.models import User, Vineyard, WeatherData
from app.models.auth import PasswordResetToken, RefreshToken
from app.services.cleanup import cleanup
from app.tasks import system
from app.worker.celery_app import celery_app

NOW = datetime(2026, 10, 29, 1, 30, tzinfo=UTC)
PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"


def test_cleanup_runs_nightly() -> None:
    schedule = celery_app.conf.beat_schedule["cleanup"]

    assert schedule["task"] == system.CLEANUP
    assert (schedule["schedule"].hour, schedule["schedule"].minute) == ({1}, {30})


def test_storage_purge_removes_only_old_job_files() -> None:
    storage = MemoryStorage()
    for key in ("jobs/u/old.jpg", "jobs/u/fresh.jpg", "photos/u/leaf.jpg"):
        storage.put(key, b"x", "image/jpeg")
    storage.modified["jobs/u/old.jpg"] = NOW - timedelta(days=2)
    storage.modified["photos/u/leaf.jpg"] = NOW - timedelta(days=30)

    assert storage.purge("jobs/", NOW - timedelta(days=1)) == 1
    assert sorted(storage.files) == ["jobs/u/fresh.jpg", "photos/u/leaf.jpg"]


def token(user: User, model: type, expires_in: timedelta, **extra: object) -> object:
    return model(
        user_id=user.id,
        token_hash=f"{model.__name__}{expires_in.total_seconds()}{extra}"[:64],
        expires_at=NOW + expires_in,
        **extra,
    )


def hour(vineyard: Vineyard, ago: timedelta, forecast: bool) -> WeatherData:
    return WeatherData(
        vineyard=vineyard,
        observed_at=NOW - ago,
        temperature_c=15,
        relative_humidity=70,
        precipitation_mm=0,
        is_forecast=forecast,
    )


@pytest.mark.integration
async def test_cleanup_removes_only_dead_records(session: AsyncSession) -> None:
    user = User(email="grower@example.md", hashed_password="hash")
    plot = Vineyard(owner=user, name="Криково", geom=WKTElement(PLOT, srid=4326))
    session.add(plot)
    await session.flush()
    session.add_all(
        [
            # Истёк два дня назад — удаляется
            token(user, RefreshToken, timedelta(days=-2)),
            # Отозван, но ещё действует: нужен для обнаружения кражи токена
            token(user, RefreshToken, timedelta(days=10), revoked_at=NOW),
            # Истёк только что — в пределах запаса
            token(user, RefreshToken, timedelta(hours=-1)),
            token(user, PasswordResetToken, timedelta(days=-3)),
            token(user, PasswordResetToken, timedelta(minutes=30)),
            hour(plot, timedelta(days=10), forecast=True),
            # Фактическая погода остаётся, как бы стара она ни была
            hour(plot, timedelta(days=11), forecast=False),
            hour(plot, timedelta(days=1), forecast=True),
        ]
    )
    await session.flush()
    storage = MemoryStorage()
    storage.put("jobs/u/lost.jpg", b"x", "image/jpeg")
    storage.modified["jobs/u/lost.jpg"] = NOW - timedelta(days=3)

    stats = await cleanup(session, storage, NOW)

    assert (
        stats.job_files,
        stats.refresh_tokens,
        stats.reset_tokens,
        stats.forecast_hours,
    ) == (1, 1, 1, 1)
    remaining = await session.scalar(select(func.count()).select_from(RefreshToken))
    assert remaining == 2
    forecasts = await session.scalars(
        select(WeatherData.is_forecast).order_by(WeatherData.observed_at)
    )
    assert forecasts.all() == [False, True]


@pytest.mark.integration
def test_cleanup_task(migrated_database_url: str) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )

    stats = asyncio.run(system.run_cleanup(settings, MemoryStorage()))

    assert stats == {
        "job_files": 0,
        "refresh_tokens": 0,
        "reset_tokens": 0,
        "forecast_hours": 0,
    }


@pytest.mark.integration
def test_cleanup_task_uses_configured_storage(
    migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    storage = MemoryStorage()
    monkeypatch.setattr(system, "get_settings", lambda: settings)
    monkeypatch.setattr(ObjectStorage, "from_settings", lambda _: storage)

    assert system.cleanup_task()["job_files"] == 0
