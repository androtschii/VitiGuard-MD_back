"""Набор данных для прогноза риска (ml/pr-014): архив погоды Open-Meteo по
винодельческим районам Молдовы, размеченный агромоделями сервера.

    uv run python -m app.cli.risk_dataset --out ml/data/raw/risk

Разметка — те же функции, что считают риск на участке (app.services.agromodels):
прогнозная модель учится предсказывать ровно то, что показывает система.
Строка — точка и местные сутки: сводка погоды и состояние моделей на этот день."""

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx2 as httpx

from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations.open_meteo import OpenMeteoClient, WeatherSample
from app.services.agromodels import HourlyWeather, label_days


@dataclass(frozen=True)
class Point:
    name: str
    region: str
    latitude: float
    longitude: float


# Три зоны с защищённым географическим указанием и север для контраста
POINTS = (
    Point("cricova", "codru", 47.14, 28.86),
    Point("ialoveni", "codru", 46.94, 28.78),
    Point("hincesti", "codru", 46.83, 28.59),
    Point("orhei", "codru", 47.38, 28.82),
    Point("purcari", "stefan_voda", 46.52, 29.86),
    Point("causeni", "stefan_voda", 46.64, 29.41),
    Point("cahul", "valul_lui_traian", 45.90, 28.19),
    Point("comrat", "valul_lui_traian", 46.30, 28.66),
    Point("taraclia", "valul_lui_traian", 45.90, 28.67),
    Point("soroca", "north", 48.16, 28.30),
)

COLUMNS = (
    "point",
    "region",
    "date",
    "mean_temp",
    "min_temp",
    "max_temp",
    "precipitation_mm",
    "mean_humidity",
    "mild_hours",
    "hours_above_35",
    "degree_days",
    # Разметка: пусто вне сезонного окна модели
    "three_tens",
    "goidanich_progress",
    "symptoms_expected",
    "gubler_thomas",
)


def label_rows(point: Point, hours: list[HourlyWeather]) -> list[dict[str, object]]:
    return [
        {"point": point.name, "region": point.region, **row}
        for row in label_days(hours)
    ]


async def history_with_retry(
    client: OpenMeteoClient,
    point: Point,
    start: date,
    end: date,
    attempts: int = 3,
    pause: float = 20.0,
) -> list[WeatherSample]:
    """Архив за 11 лет — тяжёлый запрос, и бесплатный Open-Meteo иногда
    отвечает отказом при нагрузке: несколько попыток с паузой."""
    for attempt in range(1, attempts + 1):
        try:
            return await client.history(point.latitude, point.longitude, start, end)
        except ExternalServiceError:
            if attempt == attempts:
                raise
            print(
                f"{point.name}: отказ сервиса, повтор через {pause:.0f} с", flush=True
            )
            await asyncio.sleep(pause)
    raise AssertionError  # pragma: no cover — цикл всегда возвращает или бросает


async def collect(
    client: OpenMeteoClient, points: tuple[Point, ...], start: date, end: date
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in points:
        samples = await history_with_retry(client, point, start, end)
        hours = [
            HourlyWeather(
                s.observed_at, s.temperature_c, s.relative_humidity, s.precipitation_mm
            )
            for s in samples
        ]
        point_rows = label_rows(point, hours)
        rows.extend(point_rows)
        print(f"{point.name}: {len(point_rows)} сут", flush=True)
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "daily.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ml/data/raw/risk")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args(argv)

    async def run() -> list[dict[str, object]]:
        # Архив за 11 лет по точке — несколько мегабайт и секунд
        async with httpx.AsyncClient(timeout=120) as http:
            return await collect(
                OpenMeteoClient(http, Settings(_env_file=None)),
                POINTS,
                date.fromisoformat(args.start),
                date.fromisoformat(args.end),
            )

    rows = asyncio.run(run())
    path = write_csv(rows, Path(args.out))
    print(f"Готово: {len(rows)} строк в {path}")


if __name__ == "__main__":  # pragma: no cover — запуск как скрипта
    main()
