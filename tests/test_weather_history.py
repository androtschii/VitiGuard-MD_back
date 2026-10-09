import uuid
from datetime import UTC, date, datetime
from typing import Any

import httpx2 as httpx
import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import BadRequestError, ExternalServiceError, NotFoundError
from app.integrations.open_meteo import OpenMeteoClient, WeatherSample
from app.models import User, Vineyard, WeatherData
from app.repositories.weather import WeatherRepository
from app.services.weather import import_weather_history, vineyard_point

SETTINGS = Settings(_env_file=None)
# Участок в форме буквы «С» под Криково: его центроид лежит снаружи, в «выемке»
C_SHAPE = (
    "MULTIPOLYGON(((28.860 47.140, 28.866 47.140, 28.866 47.141, 28.861 47.141,"
    " 28.861 47.145, 28.866 47.145, 28.866 47.146, 28.860 47.146, 28.860 47.140)))"
)


def archive(hours: list[str], temperature: list[float | None]) -> dict[str, Any]:
    return {
        "hourly": {
            "time": hours,
            "temperature_2m": temperature,
            "relative_humidity_2m": [80.0] * len(hours),
            "precipitation": [1.5] * len(hours),
        }
    }


def client_returning(
    payload: dict[str, Any], requests: list[httpx.Request] | None = None
) -> OpenMeteoClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, json=payload)

    return OpenMeteoClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), SETTINGS
    )


async def test_history_skips_hours_without_data() -> None:
    requests: list[httpx.Request] = []
    client = client_returning(
        archive(
            ["2026-10-01T00:00", "2026-10-01T01:00", "2026-10-01T02:00"],
            [10.0, None, 12.0],
        ),
        requests,
    )

    samples = await client.history(47.14, 28.86, date(2026, 10, 1), date(2026, 10, 1))

    assert [s.observed_at.hour for s in samples] == [0, 2]
    assert samples[0].observed_at.tzinfo is UTC
    params = requests[0].url.params
    assert str(requests[0].url).startswith(
        "https://archive-api.open-meteo.com/v1/archive"
    )
    assert params["start_date"] == params["end_date"] == "2026-10-01"
    assert params["hourly"] == "temperature_2m,relative_humidity_2m,precipitation"


async def test_series_of_different_length_are_rejected() -> None:
    payload = archive(["2026-10-01T00:00", "2026-10-01T01:00"], [10.0, 11.0])
    payload["hourly"]["precipitation"] = [0.0]

    with pytest.raises(ExternalServiceError):
        await client_returning(payload).history(
            47.14, 28.86, date(2026, 10, 1), date(2026, 10, 1)
        )


async def add_vineyard(session: AsyncSession, wkt: str = C_SHAPE) -> Vineyard:
    vineyard = Vineyard(
        owner=User(email="owner@example.md", hashed_password="hash"),
        name="Крикова",
        geom=WKTElement(wkt, srid=4326),
    )
    session.add(vineyard)
    await session.flush()
    return vineyard


@pytest.mark.integration
async def test_weather_point_is_inside_the_vineyard(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)

    latitude, longitude = await vineyard_point(session, vineyard)

    inside = await session.scalar(
        select(
            func.ST_Contains(
                Vineyard.geom,
                func.ST_SetSRID(func.ST_MakePoint(longitude, latitude), 4326),
            )
        ).where(Vineyard.id == vineyard.id)
    )
    centroid_inside = await session.scalar(
        select(func.ST_Contains(Vineyard.geom, func.ST_Centroid(Vineyard.geom))).where(
            Vineyard.id == vineyard.id
        )
    )
    assert inside is True
    assert centroid_inside is False  # поэтому и нужен ST_PointOnSurface


@pytest.mark.integration
async def test_import_stores_history_and_is_idempotent(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    hours = [f"2026-10-01T{h:02}:00" for h in range(24)]
    first = client_returning(archive(hours, [10.0] * 24))
    second = client_returning(archive(hours, [11.0] * 24))

    assert (
        await import_weather_history(
            session, first, vineyard, date(2026, 10, 1), date(2026, 10, 1)
        )
        == 24
    )
    await import_weather_history(
        session, second, vineyard, date(2026, 10, 1), date(2026, 10, 1)
    )

    rows = (await session.execute(select(WeatherData.temperature_c))).scalars().all()
    assert len(rows) == 24
    assert set(rows) == {11.0}


@pytest.mark.integration
async def test_actual_data_replaces_forecast(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    moment = datetime(2026, 10, 1, 12, tzinfo=UTC)
    repository = WeatherRepository(session)

    await repository.upsert(
        vineyard.id,
        [
            WeatherSample(
                observed_at=moment,
                temperature_c=20,
                relative_humidity=50,
                precipitation_mm=0,
            )
        ],
        is_forecast=True,
    )
    await repository.upsert(
        vineyard.id,
        [
            WeatherSample(
                observed_at=moment,
                temperature_c=18,
                relative_humidity=70,
                precipitation_mm=2,
            )
        ],
    )

    row = (
        await session.execute(
            text("SELECT temperature_c, is_forecast FROM weather_data")
        )
    ).one()
    assert tuple(row) == (18, False)


async def test_empty_upsert_does_nothing() -> None:
    class NoSession:
        async def execute(self, *_: object) -> None:
            raise AssertionError("запроса быть не должно")

    repository = WeatherRepository(NoSession())  # type: ignore[arg-type]

    assert await repository.upsert(uuid.uuid4(), []) == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (date(2026, 10, 2), date(2026, 10, 1), "Начало периода позже конца"),
        (date(2025, 1, 1), date(2026, 1, 2), "Период не длиннее 366 дней"),
    ],
)
async def test_invalid_period_is_rejected(
    session: AsyncSession, start: date, end: date, message: str
) -> None:
    vineyard = await add_vineyard(session)

    def never_called(_: httpx.Request) -> httpx.Response:
        pytest.fail("запроса быть не должно")

    client = OpenMeteoClient(
        httpx.AsyncClient(transport=httpx.MockTransport(never_called)), SETTINGS
    )

    with pytest.raises(BadRequestError, match=message):
        await import_weather_history(session, client, vineyard, start, end)


@pytest.mark.integration
async def test_point_of_deleted_vineyard_is_not_found(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    await session.execute(text("DELETE FROM vineyards"))

    with pytest.raises(NotFoundError, match="Участок не найден"):
        await vineyard_point(session, vineyard)
