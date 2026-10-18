from datetime import UTC, date, datetime, timedelta

import pytest

from app.models.enums import Disease, RiskLevel
from app.services.agromodels import (
    DailyWeather,
    HourlyWeather,
    daily_weather,
    degree_days,
    goidanich,
    gubler_thomas,
    incubation_days,
    season_risks,
    three_tens,
)


def hours(
    day: date,
    temperature: float | list[float],
    rain_mm: float = 0.0,
    humidity: float = 70.0,
) -> list[HourlyWeather]:
    """24 часа местных суток (Кишинёв летом — UTC+3): осадки за день поровну."""
    values = temperature if isinstance(temperature, list) else [temperature] * 24
    start = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(hours=3)
    return [
        HourlyWeather(start + timedelta(hours=h), values[h], humidity, rain_mm / 24)
        for h in range(24)
    ]


def day(
    when: date,
    mean: float = 15.0,
    rain: float = 0.0,
    mild_run: int = 0,
    above_35: int = 0,
) -> DailyWeather:
    return DailyWeather(when, mean, mean - 5, mean + 5, rain, 70.0, mild_run, above_35)


def season(start: date, count: int, **values: float) -> list[DailyWeather]:
    return [day(start + timedelta(days=i), **values) for i in range(count)]  # type: ignore[arg-type]


def test_daily_weather_uses_local_days() -> None:
    profile = [12.0] * 8 + [22.0] * 7 + [36.0] + [25.0] * 2 + [14.0] * 6
    result = daily_weather(hours(date(2026, 6, 1), profile, rain_mm=12))

    [summary] = result
    assert summary.day == date(2026, 6, 1)
    assert summary.precipitation_mm == pytest.approx(12)
    assert summary.min_temp == 12
    assert summary.max_temp == 36
    # 7 часов 22 °C подряд; 36 °C прерывает серию, 2 часа 25 °C — короче
    assert summary.longest_mild_run == 7
    assert summary.hours_above_35 == 1


def test_incomplete_day_is_dropped() -> None:
    assert daily_weather(hours(date(2026, 6, 1), 20)[:12]) == []


def test_degree_days_accumulate_from_march_and_reset_each_year() -> None:
    days = [
        day(date(2025, 2, 28), mean=20),
        day(date(2025, 3, 1), mean=20),
        day(date(2025, 3, 2), mean=8),
        day(date(2026, 3, 1), mean=12),
    ]

    assert degree_days(days) == {
        date(2025, 2, 28): 0,
        date(2025, 3, 1): 10,
        date(2025, 3, 2): 10,
        date(2026, 3, 1): 2,
    }


def test_three_tens_needs_warmth_rain_and_shoots() -> None:
    days = [
        *season(date(2026, 3, 1), 60, mean=13.0),  # сумма активных температур растёт
        day(date(2026, 4, 30), mean=15, rain=6),
        day(date(2026, 5, 1), mean=15, rain=5),  # 11 мм за 48 ч
        day(date(2026, 5, 2), mean=9, rain=12),  # холодно
        day(date(2026, 5, 3), mean=15, rain=0),  # вчера было 12 мм
    ]

    risks = {risk.day: risk for risk in three_tens(days)}

    assert risks[date(2026, 5, 1)].score == 1
    assert risks[date(2026, 5, 1)].level is RiskLevel.HIGH
    assert risks[date(2026, 5, 1)].disease is Disease.DOWNY_MILDEW
    assert risks[date(2026, 5, 1)].factors["rain_48h_mm"] == 11
    assert risks[date(2026, 5, 2)].score == 0
    assert risks[date(2026, 5, 3)].score == 1
    # Мартовских строк нет: до 1 апреля правило не применяется
    assert min(risks) == date(2026, 4, 1)


def test_three_tens_waits_for_shoots() -> None:
    wet = [day(date(2026, 4, 10), mean=12, rain=20)]

    assert three_tens(wet)[0].score == 0
    assert three_tens(wet)[0].factors["shoots_10_cm"] is False
    # Фенофаза по наблюдению агронома важнее оценки по температурам
    assert three_tens(wet, shoots_ready_from=date(2026, 4, 5))[0].score == 1


def test_three_tens_ignores_rain_across_a_gap() -> None:
    days = [
        day(date(2026, 5, 1), mean=15, rain=8),
        day(date(2026, 5, 3), mean=15, rain=5),
    ]

    risks = three_tens(days, shoots_ready_from=date(2026, 4, 1))

    assert [risk.factors["rain_48h_mm"] for risk in risks] == [8, 5]


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(9.9, None), (10, 20), (13, 14), (14, 12.5), (25, 4), (30, 6), (30.5, None)],
)
def test_incubation_length_follows_goidanich_curve(
    temperature: float, expected: float | None
) -> None:
    assert incubation_days(temperature) == expected


def test_goidanich_counts_incubation_to_symptoms() -> None:
    # При 25 °C инкубация — 4 дня: заражение 1 июня, пятна ожидаются 5 июня
    days = season(date(2026, 6, 1), 7, mean=25.0)

    risks = goidanich(days, [date(2026, 6, 1)])

    assert [risk.score for risk in risks] == [0, 0.25, 0.5, 0.75, 1, 0, 0]
    assert risks[3].level is RiskLevel.MEDIUM
    assert risks[4].level is RiskLevel.HIGH
    assert risks[4].factors["symptoms_expected"] is True
    assert risks[4].factors["infection_day"] == "2026-06-01"


def test_goidanich_pauses_in_cold_and_tracks_most_advanced_infection() -> None:
    days = [
        day(date(2026, 6, 1), mean=25),
        day(date(2026, 6, 2), mean=25),
        day(date(2026, 6, 3), mean=8),  # слишком холодно — развитие стоит
        day(date(2026, 6, 4), mean=25),
    ]

    risks = goidanich(days, [date(2026, 6, 1), date(2026, 6, 3)])

    assert [risk.score for risk in risks] == [0, 0.25, 0.25, 0.5]
    assert risks[3].factors["infection_day"] == "2026-06-01"
    assert risks[0].level is RiskLevel.LOW


def test_gubler_thomas_index() -> None:
    days = [
        *season(date(2026, 5, 1), 2, mild_run=6),
        day(date(2026, 5, 3)),  # серия прервана
        *season(date(2026, 5, 4), 3, mild_run=7),  # три дня подряд — 60
        day(date(2026, 5, 7), mild_run=8),  # +20
        day(date(2026, 5, 8), mild_run=8, above_35=2),  # +20 − 10
        *season(date(2026, 5, 9), 5, mild_run=1),  # −10 в день
    ]

    risks = gubler_thomas(days)
    index = [risk.factors["index"] for risk in risks]

    assert index == [0, 0, 0, 0, 0, 60, 80, 90, 80, 70, 60, 50, 40]
    assert risks[5].level is RiskLevel.HIGH
    assert risks[11].level is RiskLevel.MEDIUM
    assert risks[0].level is RiskLevel.LOW
    assert risks[5].disease is Disease.POWDERY_MILDEW
    assert risks[6].score == 0.8


def test_gubler_thomas_is_bounded_and_resets_between_seasons() -> None:
    days = [
        *season(date(2026, 5, 1), 10, mild_run=10),
        day(date(2026, 10, 1)),
        *season(date(2026, 9, 25), 1, mild_run=0, above_35=5),
    ]

    risks = gubler_thomas(sorted(days, key=lambda d: d.day))

    assert max(risk.factors["index"] for risk in risks) == 100
    assert min(risk.factors["index"] for risk in risks) >= 0
    # Октябрьский день вне сезона в ответ не попадает
    assert all(risk.day.month != 10 for risk in risks)


def test_season_risks_run_all_models() -> None:
    weather = []
    for offset in range(90):
        when = date(2026, 3, 15) + timedelta(days=offset)
        weather += hours(when, 18, rain_mm=12 if when == date(2026, 5, 20) else 0)

    risks = season_risks(weather)
    methods = {risk.method for risk in risks}

    assert methods == {"three_tens", "goidanich", "gubler_thomas"}
    infection = [r for r in risks if r.method == "three_tens" and r.score == 1]
    assert [r.day for r in infection] == [date(2026, 5, 20), date(2026, 5, 21)]
    incubation = [r for r in risks if r.method == "goidanich" and r.score > 0]
    assert incubation[0].day == date(2026, 5, 21)


def test_goidanich_keeps_daily_weather_for_the_chart() -> None:
    days = [day(date(2026, 6, 1), mean=21.5, rain=7.2)]

    factors = goidanich(days, [])[0].factors

    assert factors["mean_temp"] == 21.5
    assert factors["precipitation_mm"] == 7.2
