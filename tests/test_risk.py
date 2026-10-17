from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from geoalchemy2 import WKTElement
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_open_meteo
from app.core.errors import ExternalServiceError
from app.integrations.open_meteo import WeatherSample
from app.models import DiseaseRisk, User, Vineyard, WeatherData
from app.services.risk import (
    local_today,
    merge_weather,
    refresh_risks,
    season_risk_history,
    season_year,
)

SQUARE = (
    "MULTIPOLYGON(((28.860 47.140, 28.862 47.140, 28.862 47.141, 28.860 47.141,"
    " 28.860 47.140)))"
)
# Дождь 20 мая — первичное заражение милдью, если побеги уже отросли
RAIN_DAY = (5, 20)


def sample(moment: datetime, temperature: float, rain: float = 0.0) -> WeatherSample:
    return WeatherSample(
        observed_at=moment,
        temperature_c=temperature,
        relative_humidity=75.0,
        precipitation_mm=rain,
    )


class FakeOpenMeteo:
    """Архив: 18 °C, 20 мая — 12 мм за сутки. Прогноз: 25 °C без осадков."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def history(
        self, _lat: float, _lon: float, start: date, end: date
    ) -> list[WeatherSample]:
        self.calls.append(f"history {start} {end}")
        if self.fail:
            raise ExternalServiceError("Сервис погоды недоступен")
        moment = datetime(start.year, start.month, start.day, tzinfo=UTC)
        stop = datetime(end.year, end.month, end.day, 23, tzinfo=UTC)
        samples = []
        while moment <= stop:
            wet = (moment.month, moment.day) == RAIN_DAY
            samples.append(sample(moment, 18.0, 0.5 if wet else 0.0))
            moment += timedelta(hours=1)
        return samples

    async def forecast(
        self, _lat: float, _lon: float, past_days: int, forecast_days: int
    ) -> list[WeatherSample]:
        self.calls.append(f"forecast {past_days} {forecast_days}")
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        return [
            sample(now + timedelta(hours=h), 25.0)
            for h in range(-24 * past_days, 24 * forecast_days)
        ]


class FrozenForecast(FakeOpenMeteo):
    """Прогноз относительно заданного «сейчас»."""

    def __init__(self, now: datetime) -> None:
        super().__init__()
        self.now = now

    async def forecast(
        self, _lat: float, _lon: float, past_days: int, forecast_days: int
    ) -> list[WeatherSample]:
        hour = self.now.replace(minute=0, second=0, microsecond=0)
        return [
            sample(hour + timedelta(hours=h), 25.0)
            for h in range(-24 * past_days, 24 * forecast_days)
        ]


async def add_vineyard(session: AsyncSession) -> Vineyard:
    vineyard = Vineyard(
        owner=User(email="owner@example.md", hashed_password="hash"),
        name="Криково",
        geom=WKTElement(SQUARE, srid=4326),
    )
    session.add(vineyard)
    await session.flush()
    return vineyard


def test_season_year_and_local_day() -> None:
    assert season_year(date(2026, 10, 17)) == 2026
    assert season_year(date(2027, 2, 10)) == 2026
    # 22:30 UTC 31 мая — в Кишинёве уже 1 июня
    assert local_today(datetime(2026, 5, 31, 22, 30, tzinfo=UTC)) == date(2026, 6, 1)


def test_merge_prefers_archive_and_sorts() -> None:
    at = datetime(2026, 6, 1, tzinfo=UTC)
    archive = [sample(at, 18.0)]
    recent = [sample(at + timedelta(hours=1), 25.0), sample(at, 30.0)]

    merged = merge_weather(archive, recent)

    assert [s.temperature_c for s in merged] == [18.0, 25.0]


@pytest.mark.integration
async def test_refresh_saves_weather_and_daily_risks(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    now = datetime(2026, 6, 10, 9, tzinfo=UTC)
    client = FrozenForecast(now)

    saved = await refresh_risks(session, client, vineyard, now)  # type: ignore[arg-type]

    # Архив — с 1 марта до недели назад, дальше — сервис прогноза
    assert client.calls == ["history 2026-03-01 2026-06-03"]
    weather = await session.scalar(
        select(func.count()).select_from(WeatherData).where(WeatherData.is_forecast)
    )
    assert weather == 7 * 24 - 1
    rows = await season_risk_history(session, vineyard, date(2026, 6, 10))
    assert len(rows) == saved
    by_method = {(r.method, r.target_date): r for r in rows}
    assert by_method[("three_tens", date(2026, 5, 20))].score == 1
    incubation = by_method[("goidanich", date(2026, 5, 22))]
    assert 0 < incubation.score < 1
    assert incubation.factors is not None
    assert incubation.factors["infection_day"] == "2026-05-20"
    # Дни после сегодняшнего — по прогнозу погоды
    ahead = by_method[("gubler_thomas", date(2026, 6, 15))].factors
    past = by_method[("gubler_thomas", date(2026, 6, 9))].factors
    assert ahead is not None
    assert past is not None
    assert ahead["forecast"] is True
    assert past["forecast"] is False

    # Повторный пересчёт обновляет строки, а не дублирует
    again = await refresh_risks(session, client, vineyard, now)  # type: ignore[arg-type]
    total = await session.scalar(select(func.count()).select_from(DiseaseRisk))
    assert total == again


@pytest.mark.integration
async def test_early_spring_has_no_archive_yet(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    now = datetime(2026, 3, 3, 9, tzinfo=UTC)
    client = FrozenForecast(now)

    saved = await refresh_risks(session, client, vineyard, now)  # type: ignore[arg-type]

    # 3 марта архива сезона ещё нет, а риск считается только с апреля
    assert client.calls == []
    assert saved == 0


@pytest.mark.integration
async def test_winter_shows_last_season_from_archive(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    client = FakeOpenMeteo()

    saved = await refresh_risks(
        session,
        client,  # type: ignore[arg-type]
        vineyard,
        datetime(2027, 1, 20, 9, tzinfo=UTC),
    )

    # Сезон 2026 года целиком из архива, прогноз прошлому сезону не нужен
    assert client.calls == ["history 2026-03-01 2026-12-31"]
    assert saved > 0


CREDENTIALS = {"email": "grower@example.md", "password": "secret123"}


def login(client: TestClient, email: str = CREDENTIALS["email"]) -> dict[str, str]:
    credentials = {**CREDENTIALS, "email": email}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def use_weather(client: TestClient, fake: FakeOpenMeteo) -> None:
    client.app.dependency_overrides[get_open_meteo] = lambda: fake  # type: ignore[attr-defined]


def create_vineyard(client: TestClient, headers: dict[str, str]) -> str:
    geometry: dict[str, Any] = {
        "type": "Polygon",
        "coordinates": [
            [[28.86, 47.14], [28.862, 47.14], [28.862, 47.141], [28.86, 47.14]]
        ],
    }
    response = client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": geometry},
    )
    vineyard_id: str = response.json()["id"]
    return vineyard_id


@pytest.mark.integration
def test_risk_api(api_client: TestClient) -> None:
    use_weather(api_client, FakeOpenMeteo())
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)
    url = f"/api/v1/vineyards/{vineyard_id}/risks"

    assert api_client.get(url, headers=headers).json() == []
    refreshed = api_client.post(f"{url}/refresh", headers=headers)
    stored = api_client.get(url, headers=headers).json()

    assert refreshed.status_code == 200
    assert refreshed.json() == stored
    assert {item["method"] for item in stored} == {
        "three_tens",
        "goidanich",
        "gubler_thomas",
    }
    first = stored[0]
    assert set(first) == {
        "target_date",
        "disease",
        "method",
        "score",
        "level",
        "factors",
        "updated_at",
    }
    api_client.app.dependency_overrides.clear()  # type: ignore[attr-defined]


@pytest.mark.integration
def test_risk_api_errors(api_client: TestClient) -> None:
    use_weather(api_client, FakeOpenMeteo(fail=True))
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)
    stranger = login(api_client, "stranger@example.md")
    url = f"/api/v1/vineyards/{vineyard_id}/risks"

    failed = api_client.post(f"{url}/refresh", headers=headers)

    assert failed.status_code == 502
    assert failed.json() == {"detail": "Сервис погоды недоступен"}
    assert api_client.get(url, headers=stranger).status_code == 404
    assert api_client.post(f"{url}/refresh", headers=stranger).status_code == 404
    assert api_client.get(url).status_code == 401
    api_client.app.dependency_overrides.clear()  # type: ignore[attr-defined]
