"""Ряды NDVI и NDRE по виноградникам для поиска аномалий (ml/pr-019).

    uv run python -m app.cli.ndvi_dataset --vineyards ml/data/raw/ndvi/vineyards.geojson

Для каждого виноградника из GeoJSON (свойства id и point) и каждого сезона —
снимки Sentinel-2 из Earth Search и индексы по пикселям участка без облаков,
теми же функциями, что считает сервер. Снимки обрабатываются параллельно:
чтение окон по HTTP — ожидание сети, а не процессора."""

import argparse
import asyncio
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2 as httpx

from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations.earth_search import (
    EarthScene,
    EarthSearchClient,
    cloud_query,
    deduplicate,
)
from app.services.vegetation import compute_indices, read_parcel

COLUMNS = (
    "vineyard",
    "point",
    "area_ha",
    "date",
    "platform",
    "scene_cloud",
    "parcel_cloud",
    "parcel_pixels",
    "clear_pixels",
    "ndvi_mean",
    "ndvi_std",
    "ndvi_min",
    "ndvi_max",
    "ndre_mean",
    "ndre_std",
)


def observation(feature: dict[str, Any], scene: EarthScene) -> dict[str, object] | None:
    """Строка ряда по снимку; None — каналы не прочитались (сбой хранилища)."""
    try:
        result = compute_indices(read_parcel(scene, feature["geometry"]))
    except OSError:
        return None
    ndvi, ndre = result.ndvi, result.ndre
    properties = feature["properties"]
    return {
        "vineyard": properties["id"],
        "point": properties.get("point", ""),
        "area_ha": properties.get("area_ha", ""),
        "date": scene.acquired_at.date().isoformat(),
        "platform": scene.platform,
        "scene_cloud": round(scene.cloud_cover, 2),
        "parcel_cloud": round(result.cloud_share * 100, 1),
        "parcel_pixels": result.parcel_pixels,
        "clear_pixels": result.clear_pixels,
        "ndvi_mean": ndvi.mean if ndvi else "",
        "ndvi_std": ndvi.std if ndvi else "",
        "ndvi_min": ndvi.min if ndvi else "",
        "ndvi_max": ndvi.max if ndvi else "",
        "ndre_mean": ndre.mean if ndre else "",
        "ndre_std": ndre.std if ndre else "",
    }


async def collect(
    client: EarthSearchClient,
    features: list[dict[str, Any]],
    seasons: list[tuple[datetime, datetime]],
    max_cloud: float,
    workers: int,
) -> tuple[list[dict[str, object]], int]:
    """Строки по всем виноградникам и сезонам и число снимков со сбоем."""
    rows: list[dict[str, object]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for feature in features:
            scenes: list[EarthScene] = []
            for start, end in seasons:
                scenes += deduplicate(
                    await client.search(
                        feature["geometry"],
                        start,
                        end,
                        extra_body=cloud_query(max_cloud),
                    )
                )
            results = await asyncio.gather(
                *(
                    asyncio.get_running_loop().run_in_executor(
                        pool, observation, feature, scene
                    )
                    for scene in scenes
                )
            )
            done = [row for row in results if row is not None]
            failed += len(results) - len(done)
            rows += done
            print(f"{feature['properties']['id']}: {len(done)} снимков", flush=True)
    return rows, failed


async def collect_with_retries(
    client: EarthSearchClient,
    feature: dict[str, Any],
    seasons: list[tuple[datetime, datetime]],
    max_cloud: float,
    workers: int,
    attempts: int = 4,
    pause: float = 30,
) -> tuple[list[dict[str, object]], int]:
    """Каталог иногда отвечает ошибкой на несколько минут: виноградник
    повторяется после паузы, а не обрывает многочасовой сбор."""
    for attempt in range(1, attempts + 1):
        try:
            return await collect(client, [feature], seasons, max_cloud, workers)
        except ExternalServiceError:
            if attempt == attempts:
                raise
            await asyncio.sleep(pause * attempt)
    raise AssertionError  # pragma: no cover — цикл всегда возвращает или падает


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda r: (str(r["vineyard"]), str(r["date"])))
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vineyards", default="ml/data/raw/ndvi/vineyards.geojson")
    parser.add_argument("--out", default="ml/data/raw/ndvi/observations.csv")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--max-cloud", type=float, default=40)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    features = json.loads(Path(args.vineyards).read_text(encoding="utf-8"))["features"]
    # Вегетация — с апреля по середину октября
    seasons = [
        (datetime(year, 4, 1, tzinfo=UTC), datetime(year, 10, 15, 23, 59, tzinfo=UTC))
        for year in args.seasons
    ]
    out = Path(args.out)
    # Сбор идёт часами: файл пишется после каждого виноградника, а при
    # повторном запуске собранные виноградники пропускаются
    rows = read_csv(out)
    done = {str(row["vineyard"]) for row in rows}
    pending = [f for f in features if str(f["properties"]["id"]) not in done]
    failed = 0

    async def run() -> None:
        nonlocal failed
        async with httpx.AsyncClient(timeout=60) as http:
            client = EarthSearchClient(http, Settings(_env_file=None))
            for feature in pending:
                found, errors = await collect_with_retries(
                    client, feature, seasons, args.max_cloud, args.workers
                )
                rows.extend(found)
                failed += errors
                write_csv(rows, out)

    asyncio.run(run())
    print(f"Готово: {len(rows)} снимков, сбоев чтения {failed}, файл {args.out}")


if __name__ == "__main__":  # pragma: no cover — запуск как скрипта
    main()
