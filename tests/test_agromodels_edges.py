"""Граничные случаи и инварианты агромоделей (pr-085): переход на летнее и
зимнее время, точные пороги правил, края сезонных окон и свойства, которые
должны выполняться на любой погоде."""

import random
from collections import Counter
from datetime import UTC, date, datetime, timedelta

import pytest

from app.models.enums import RiskLevel
from app.services.agromodels import (
    DailyWeather,
    HourlyWeather,
    daily_weather,
    goidanich,
    gubler_thomas,
    season_risks,
    three_tens,
)


def utc_hours(start: datetime, count: int, rain: float = 1.0) -> list[HourlyWeather]:
    return [
        HourlyWeather(start + timedelta(hours=h), 15.0, 80.0, rain)
        for h in range(count)
    ]


def day(when: date, **values: float) -> DailyWeather:
    defaults: dict[str, float] = {
        "mean_temp": 15.0,
        "min_temp": 10.0,
        "max_temp": 20.0,
        "precipitation_mm": 0.0,
        "mean_humidity": 70.0,
        "longest_mild_run": 0,
        "hours_above_35": 0,
    }
    defaults.update(values)
    return DailyWeather(when, **defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("local_day", "hours"),
    [
        # 29 марта 2026 часы переводятся вперёд: в местных сутках 23 часа
        (date(2026, 3, 29), 23),
        # 25 октября 2026 — назад: 25 часов
        (date(2026, 10, 25), 25),
        (date(2026, 6, 1), 24),
    ],
)
def test_local_day_follows_daylight_saving(local_day: date, hours: int) -> None:
    # По миллиметру в час двое суток вокруг перевода часов
    start = datetime(local_day.year, local_day.month, local_day.day, tzinfo=UTC)
    weather = utc_hours(start - timedelta(days=1), 72)

    summary = {item.day: item for item in daily_weather(weather)}[local_day]

    assert summary.precipitation_mm == pytest.approx(hours)


def test_three_tens_thresholds_are_inclusive() -> None:
    shoots = date(2026, 4, 1)

    def infected(mean: float, rain: float) -> bool:
        [risk] = three_tens(
            [day(date(2026, 5, 10), mean_temp=mean, precipitation_mm=rain)], shoots
        )
        return risk.score == 1

    assert infected(10.0, 10.0)
    assert not infected(9.99, 10.0)
    assert not infected(10.0, 9.99)


def test_season_windows_edges() -> None:
    edges = [
        date(2026, 3, 31),
        date(2026, 4, 1),
        date(2026, 6, 30),
        date(2026, 7, 1),
        date(2026, 9, 30),
        date(2026, 10, 1),
    ]
    days = [day(when) for when in edges]

    assert [risk.day for risk in three_tens(days)] == [
        date(2026, 4, 1),
        date(2026, 6, 30),
    ]
    inside = [date(2026, 4, 1), date(2026, 6, 30), date(2026, 7, 1), date(2026, 9, 30)]
    assert [risk.day for risk in goidanich(days, [])] == inside
    assert [risk.day for risk in gubler_thomas(days)] == inside


def test_gubler_thomas_level_boundaries() -> None:
    start = date(2026, 6, 1)
    days = [
        *(day(start + timedelta(days=i), longest_mild_run=6) for i in range(3)),
        *(day(start + timedelta(days=3 + i)) for i in range(3)),
    ]

    risks = gubler_thomas(days)

    assert [(r.factors["index"], r.level) for r in risks[2:]] == [
        (60, RiskLevel.HIGH),
        (50, RiskLevel.MEDIUM),
        (40, RiskLevel.MEDIUM),
        (30, RiskLevel.LOW),
    ]


def test_gubler_thomas_floor_keeps_epidemic_started() -> None:
    start = date(2026, 6, 1)
    days = [
        *(day(start + timedelta(days=i), longest_mild_run=8) for i in range(3)),
        # Жаркие дни без мягких часов: −20 в день, но не ниже нуля
        *(day(start + timedelta(days=3 + i), hours_above_35=4) for i in range(4)),
        # Отсчёт уже начался: один мягкий день сразу даёт +20, без новых трёх
        day(start + timedelta(days=7), longest_mild_run=8),
    ]

    index = [risk.factors["index"] for risk in gubler_thomas(days)]

    assert index == [0, 0, 60, 40, 20, 0, 0, 20]


def test_goidanich_follows_each_infection_to_its_symptoms() -> None:
    # Заражения 1 и 2 июня (порядок в списке не важен): при 25 °C у каждого
    # свои четыре дня инкубации, и пятна ожидаются 5-го и 6-го
    days = [day(date(2026, 6, d), mean_temp=25.0) for d in range(1, 8)]

    risks = goidanich(days, [date(2026, 6, 2), date(2026, 6, 1)])
    finished = [risk for risk in risks if risk.factors.get("symptoms_expected")]

    assert [risk.factors["infection_day"] for risk in finished] == [
        "2026-06-01",
        "2026-06-02",
    ]


def random_season(seed: int, year: int) -> list[HourlyWeather]:
    rng = random.Random(seed)
    start = datetime(year, 3, 1, tzinfo=UTC)
    # Каждый пятый день дождливый: от 2 до 20 мм за сутки
    wet = {offset: rng.uniform(2, 20) for offset in range(214) if rng.random() < 0.2}
    weather = []
    for hour in range(24 * 214):
        moment = start + timedelta(hours=hour)
        seasonal = 8 + 14 * min(1.0, hour / (24 * 120))
        daily = 6 * ((moment.hour - 3) % 24 < 12)
        temperature = seasonal + daily + rng.gauss(0, 3)
        rain = wet.get(hour // 24, 0.0) / 24
        weather.append(HourlyWeather(moment, temperature, rng.uniform(40, 100), rain))
    return weather


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_invariants_on_random_weather(seed: int) -> None:
    weather = random_season(seed, 2024 + seed)

    risks = season_risks(weather)

    assert risks == season_risks(weather)
    # В сезоне есть и заражения, и конец инкубации: проверки ниже не пустые
    assert any(r.method == "three_tens" and r.score == 1 for r in risks)
    assert any(r.method == "goidanich" and r.score == 1 for r in risks)
    assert all(0 <= risk.score <= 1 for risk in risks)
    per_day = Counter((risk.method, risk.day) for risk in risks)
    assert max(per_day.values()) == 1
    for risk in risks:
        if risk.method == "three_tens":
            assert 4 <= risk.day.month <= 6
            assert (risk.score == 1) == (risk.level is RiskLevel.HIGH)
        elif risk.method == "gubler_thomas":
            assert risk.factors["index"] % 10 == 0
            assert 0 <= risk.factors["index"] <= 100
        else:
            assert 4 <= risk.day.month <= 9
            # Конец инкубации — только после заражения по «трём десяткам»
            if risk.score == 1:
                assert date.fromisoformat(risk.factors["infection_day"]) < risk.day
