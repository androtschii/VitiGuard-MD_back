from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from test_risk import (
    FrozenForecast,
    add_vineyard,
    create_vineyard,
    login,
    sample,
    use_weather,
)

from app.integrations.open_meteo import WeatherSample
from app.services.agromodels import HourlyWeather
from app.services.outlook import outlook_from_members, vineyard_outlook

TODAY = date(2026, 5, 10)
START = datetime(2026, 3, 1, tzinfo=UTC)
NOW = datetime(2026, 5, 10, 9, tzinfo=UTC)


def hours(
    start: datetime, end: datetime, temperature: float, wet: date | None = None
) -> list[HourlyWeather]:
    """Ровная погода; в день wet за сутки выпадает 12 мм."""
    result = []
    moment = start
    while moment < end:
        rain = 0.5 if wet and moment.date() == wet else 0.0
        result.append(HourlyWeather(moment, temperature, 75.0, rain))
        moment += timedelta(hours=1)
    return result


def test_probability_is_share_of_members_with_outbreak() -> None:
    # С марта 18 °C: побеги давно отросли, оидиум при такой погоде не начинается
    history = hours(START, NOW, 18.0)
    end = NOW + timedelta(days=8)
    rainy = hours(NOW - timedelta(days=2), end, 18.0, wet=date(2026, 5, 13))
    dry = hours(NOW - timedelta(days=2), end, 18.0)

    outlook = outlook_from_members(history, [rainy, dry, dry, dry], TODAY)

    assert outlook.members == 4
    assert [day.day for day in outlook.days] == [
        TODAY + timedelta(days=offset) for offset in range(1, 8)
    ]
    downy = {day.day: day.downy for day in outlook.days}
    # Дождь 13 мая у одного из четырёх: заражение 13-го и 14-го (осадки за 48 ч)
    assert downy[date(2026, 5, 13)] == downy[date(2026, 5, 14)] == 0.25
    assert downy[date(2026, 5, 12)] == downy[date(2026, 5, 15)] == 0
    assert outlook.downy_week == 0.25
    assert all(day.powdery == 0 for day in outlook.days)
    assert outlook.powdery_week == 0


def test_history_wins_over_members_in_shared_hours() -> None:
    # Члены ансамбля начинаются за двое суток до «сейчас» с ливнем вчера, но
    # вчерашний день уже известен по факту — сухой
    history = hours(START, NOW, 18.0)
    members = [
        hours(
            NOW - timedelta(days=2), NOW + timedelta(days=8), 18.0, wet=date(2026, 5, 9)
        )
    ]

    outlook = outlook_from_members(history, members, TODAY)

    assert outlook.downy_week == 0


def test_hot_members_raise_powdery_risk() -> None:
    history = hours(START, NOW, 18.0)
    end = NOW + timedelta(days=8)
    hot = hours(NOW, end, 25.0)
    mild = hours(NOW, end, 18.0)

    outlook = outlook_from_members(history, [hot, mild], TODAY)

    powdery = [day.powdery for day in outlook.days]
    # Жара с утра сегодня, и сегодняшний день — первый из трёх жарких подряд:
    # послезавтра индекс начинается с 60 баллов
    assert powdery == [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert outlook.powdery_week == 0.5


def test_outside_season_there_is_nothing_to_forecast() -> None:
    outlook = outlook_from_members([], [], date(2026, 11, 10))

    assert outlook.members == 0
    assert all(day.downy is None and day.powdery is None for day in outlook.days)
    assert outlook.downy_week is None
    assert outlook.powdery_week is None


def test_primary_infection_window_closes_in_june() -> None:
    today = date(2026, 6, 27)
    now = datetime(2026, 6, 27, 9, tzinfo=UTC)
    rainy = hours(now, now + timedelta(days=8), 18.0, wet=date(2026, 7, 2))

    outlook = outlook_from_members(hours(START, now, 18.0), [rainy], today)

    # После 30 июня «три десятки» не считаются, но инкубация и оидиум — да
    july = {day.day: day.downy for day in outlook.days}[date(2026, 7, 2)]
    assert july == 0


class EnsembleWeather(FrozenForecast):
    """Ансамбль из четырёх членов: два жарких и сухих, два прохладных с ливнем
    13 июня."""

    def __init__(self, now: datetime) -> None:
        super().__init__(now)
        self.calls: list[str] = []

    async def ensemble(
        self, _lat: float, _lon: float, past_days: int, forecast_days: int
    ) -> list[list[WeatherSample]]:
        self.calls.append(f"ensemble {past_days} {forecast_days}")
        hour = self.now.replace(minute=0, second=0, microsecond=0)
        moments = [
            hour + timedelta(hours=h)
            for h in range(-24 * past_days, 24 * forecast_days)
        ]
        hot = [sample(moment, 25.0) for moment in moments]
        rainy = [
            sample(moment, 15.0, 0.5 if moment.date() == date(2026, 6, 13) else 0.0)
            for moment in moments
        ]
        return [hot, hot, rainy, rainy]


@pytest.mark.integration
async def test_outlook_continues_season_weather(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    now = datetime(2026, 6, 10, 9, tzinfo=UTC)
    client = EnsembleWeather(now)

    outlook = await vineyard_outlook(session, client, vineyard, now)  # type: ignore[arg-type]

    # Погоды сезона ещё нет — сначала обычный пересчёт риска
    assert client.calls == ["history 2026-03-01 2026-06-03", "ensemble 2 8"]
    assert outlook.members == 4
    by_day = {day.day: day for day in outlook.days}
    assert by_day[date(2026, 6, 13)].downy == by_day[date(2026, 6, 14)].downy == 0.5
    assert by_day[date(2026, 6, 12)].downy == 0
    # Неделя жары перед сегодняшним днём довела индекс оидиума до 100: в
    # прохладных вариантах он падает на 10 в день и с 15 июня ниже 60
    assert [by_day[date(2026, 6, d)].powdery for d in range(11, 18)] == [
        1,
        1,
        1,
        1,
        0.5,
        0.5,
        0.5,
    ]
    assert outlook.downy_week == 0.5
    assert outlook.powdery_week == 1

    # Погода сохранена: второй раз пересчёт не нужен
    client.calls.clear()
    await vineyard_outlook(session, client, vineyard, now)  # type: ignore[arg-type]
    assert client.calls == ["ensemble 2 8"]


@pytest.mark.integration
def test_outlook_api(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api.routes import risks
    from app.services import outlook

    monkeypatch.setattr(risks, "local_today", lambda: date(2026, 11, 10))
    monkeypatch.setattr(outlook, "local_today", lambda _=None: date(2026, 11, 10))
    computed: list[int] = []
    original = vineyard_outlook

    async def counted(*args: Any) -> Any:
        computed.append(1)
        return await original(*args)

    monkeypatch.setattr(risks, "vineyard_outlook", counted)
    use_weather(api_client, EnsembleWeather(datetime(2026, 11, 10, tzinfo=UTC)))
    headers = login(api_client)
    vineyard_id = create_vineyard(api_client, headers)
    url = f"/api/v1/vineyards/{vineyard_id}/risks/outlook"

    first = api_client.get(url, headers=headers)
    second = api_client.get(url, headers=headers)

    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"members", "days", "downy_week", "powdery_week"}
    # В ноябре моделям нечего оценивать
    assert body["members"] == 0
    assert body["days"][0] == {"day": "2026-11-11", "downy": None, "powdery": None}
    assert second.json() == body
    assert len(computed) == 1

    stranger = login(api_client, "stranger@example.md")
    assert api_client.get(url, headers=stranger).status_code == 404
    assert api_client.get(url).status_code == 401
    api_client.app.dependency_overrides.clear()  # type: ignore[attr-defined]
