"""Симуляция сезона для лабораторных работ (pr-099): правдоподобная погода и
спутниковые снимки участка без интернета и без ожидания реального сезона.

Ряды строятся из зерна: у каждого студента свой сезон, но тот же самый при
повторном запуске — результат лабораторной можно проверить. Сценарии:

- dry — сухой сезон: редкие дожди, заражений милдью почти нет, оидиум в жару;
- normal — обычный для Молдовы сезон;
- wet — дождливый: частые первичные заражения милдью;
- outbreak — дождливый сезон с очагом болезни: в середине июля NDVI участка
  проседает на два-три снимка подряд, и детектор аномалий должен это найти.

Погода — почасовая, в UTC, как от Open-Meteo, поэтому дальше работают те же
агромодели, что и на настоящих данных."""

import math
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

from app.services.agromodels import LOCAL_TZ, HourlyWeather


class Scenario(StrEnum):
    dry = "dry"
    normal = "normal"
    wet = "wet"
    outbreak = "outbreak"


# Доля дождливых дней и средний дождь за такой день, мм
RAIN = {
    Scenario.dry: (0.10, 5.0),
    Scenario.normal: (0.20, 7.0),
    Scenario.wet: (0.32, 10.0),
    Scenario.outbreak: (0.32, 10.0),
}
# Жаркий сухой сезон теплее на пару градусов
WARMER = {Scenario.dry: 2.0}

SEASON = ((3, 1), (9, 30))
SCAN_EVERY_DAYS = 5
# Очаг в сценарии outbreak: с этого дня NDVI ниже нормы на OUTBREAK_DROP
OUTBREAK_START = (7, 12)
OUTBREAK_DROP = 0.2
OUTBREAK_SCANS = 3


@dataclass(frozen=True)
class SimulatedScan:
    acquired_at: datetime
    # Облака над участком, %; 100 — снимок закрыт, индексов нет
    parcel_cloud_cover: float
    scene_cloud_cover: float
    ndvi: float | None
    ndre: float | None


def _season_days(year: int) -> list[date]:
    (m1, d1), (m2, d2) = SEASON
    first, last = date(year, m1, d1), date(year, m2, d2)
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def mean_temperature(day: date) -> float:
    """Климатическая среднесуточная температура Кодр: около 5 °C в начале
    марта, 23-24 °C в конце июля."""
    doy = day.timetuple().tm_yday
    return 11.5 - 12.5 * math.cos(2 * math.pi * (doy - 22) / 365.25)


@dataclass(frozen=True)
class _Day:
    mean: float
    rainy: bool
    rain_total: float
    rain_start: int
    rain_hours: int


def simulate_weather(year: int, scenario: Scenario, seed: int) -> list[HourlyWeather]:
    rng = random.Random(f"weather-{year}-{scenario}-{seed}")
    chance, mean_rain = RAIN[scenario]
    anomaly = 0.0
    days: dict[date, _Day] = {}
    for day in _season_days(year):
        # Погода держится несколько дней: отклонение дня — авторегрессия
        anomaly = 0.7 * anomaly + rng.gauss(0, 1.8)
        rainy = rng.random() < chance
        days[day] = _Day(
            mean=mean_temperature(day) + WARMER.get(scenario, 0.0) + anomaly,
            rainy=rainy,
            rain_total=rng.expovariate(1 / mean_rain) if rainy else 0.0,
            # Дождь идёт днём, несколько часов подряд
            rain_start=rng.randint(11, 17),
            rain_hours=rng.randint(3, 7),
        )
    # Шаг — час UTC, местное время выводится из него: в дни перевода часов в
    # сутках 23 или 25 часов, и ни один час не повторяется
    first = min(days)
    moment = datetime(first.year, first.month, first.day, tzinfo=LOCAL_TZ).astimezone(
        UTC
    )
    hours: list[HourlyWeather] = []
    while (local := moment.astimezone(LOCAL_TZ)).date() in days:
        today = days[local.date()]
        # Суточный ход: минимум перед рассветом, максимум около 15 часов; в
        # дождливый день размах меньше
        swing = (3.5 if today.rainy else 6.0) * math.cos(
            2 * math.pi * (local.hour - 15) / 24
        )
        temperature = today.mean + swing + rng.gauss(0, 0.4)
        raining = today.rainy and (
            today.rain_start <= local.hour < today.rain_start + today.rain_hours
        )
        humidity = min(
            100.0, max(25.0, 95 - 2.2 * (temperature - 8) + (15 if raining else 0))
        )
        hours.append(
            HourlyWeather(
                moment,
                round(temperature, 1),
                round(humidity, 0),
                round(today.rain_total / today.rain_hours, 2) if raining else 0.0,
            )
        )
        moment += timedelta(hours=1)
    return hours


def ndvi_norm(day: date) -> float:
    """NDVI здорового участка: распускание с середины апреля и рост почти два
    месяца, как в Молдове; плато в июле, медленное старение листа с августа
    к концу сентября."""
    doy = day.timetuple().tm_yday
    green = 0.24 + 0.48 / (1 + math.exp(-(doy - 140) / 13))
    ageing = 0.0015 * max(0, doy - 215)
    return green - ageing


def simulate_scans(year: int, scenario: Scenario, seed: int) -> list[SimulatedScan]:
    rng = random.Random(f"scans-{year}-{scenario}-{seed}")
    outbreak = date(year, *OUTBREAK_START)
    dropped = 0
    scans = []
    day = date(year, 4, 1)
    while day <= date(year, 9, 30):
        scene = round(rng.uniform(0, 40), 1)
        cloudy = rng.random() < 0.2
        if cloudy:
            scans.append(SimulatedScan(_pass_time(day), 100.0, scene, None, None))
        else:
            ndvi = ndvi_norm(day) + rng.gauss(0, 0.008)
            if (
                scenario is Scenario.outbreak
                and day >= outbreak
                and dropped < OUTBREAK_SCANS
            ):
                ndvi -= OUTBREAK_DROP
                dropped += 1
            ndre = 0.62 * ndvi - 0.06 + rng.gauss(0, 0.005)
            scans.append(
                SimulatedScan(
                    _pass_time(day),
                    round(rng.uniform(0, 8), 1),
                    scene,
                    round(ndvi, 4),
                    round(ndre, 4),
                )
            )
        day += timedelta(days=SCAN_EVERY_DAYS)
    return scans


def _pass_time(day: date) -> datetime:
    # Sentinel-2 пролетает над Молдовой около 9:17 UTC
    return datetime(day.year, day.month, day.day, 9, 17, tzinfo=UTC)
