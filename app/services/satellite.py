import json
import re
from datetime import datetime

from geoalchemy2 import functions as geo
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import BadRequestError, NotFoundError
from app.integrations.copernicus import CopernicusClient, Scene
from app.models.vineyard import Vineyard

# Версия обработки в имени продукта: S2A_MSIL2A_20261008T091031_N0513_R050_...
BASELINE = re.compile(r"_N(\d{4})_")


def _baseline(scene: Scene) -> int:
    match = BASELINE.search(scene.product_id)
    return int(match.group(1)) if match else 0


def cloud_filter(max_cloud_cover: float) -> dict[str, object]:
    """CQL2-фильтр «облачность сцены не выше порога» — каталог применяет его сам,
    и облачные снимки даже не передаются по сети."""
    return {"op": "<=", "args": [{"property": "eo:cloud_cover"}, max_cloud_cover]}


def deduplicate(scenes: list[Scene]) -> list[Scene]:
    """Одна съёмка — один снимок.

    Одна и та же съёмка попадает в каталог несколько раз: участок на стыке
    тайлов MGRS виден в двух тайлах, а старые снимки ESA переобрабатывает новой
    версией обработки. Из повторов остаётся снимок новейшей обработки, затем — с
    меньшей облачностью; результат упорядочен по дате съёмки."""
    best: dict[tuple[str, datetime], Scene] = {}
    for scene in scenes:
        key = (scene.platform, scene.acquired_at.replace(microsecond=0))
        current = best.get(key)
        rank = (_baseline(scene), -scene.cloud_cover, scene.tile)
        if current is None or rank > (
            _baseline(current),
            -current.cloud_cover,
            current.tile,
        ):
            best[key] = scene
    return sorted(best.values(), key=lambda scene: scene.acquired_at)


async def vineyard_geojson(
    session: AsyncSession, vineyard: Vineyard
) -> dict[str, object]:
    geojson = await session.scalar(
        select(geo.ST_AsGeoJSON(Vineyard.geom)).where(Vineyard.id == vineyard.id)
    )
    if geojson is None:
        raise NotFoundError("Участок не найден")
    parsed: dict[str, object] = json.loads(geojson)
    return parsed


async def find_scenes(
    session: AsyncSession,
    copernicus: CopernicusClient,
    vineyard: Vineyard,
    start: datetime,
    end: datetime,
    settings: Settings,
    max_cloud_cover: float | None = None,
) -> list[Scene]:
    """Снимки Sentinel-2 над участком за период, без повторов и без облачных.

    Облачность — по всей сцене (квадрат 110 × 110 км): над самим участком
    облаков может не быть и при облачной сцене, и наоборот. Точная проверка по
    маске SCL в границах участка делается после загрузки каналов (pr-062)."""
    if start >= end:
        raise BadRequestError("Начало периода должно быть раньше конца")
    threshold = (
        settings.max_scene_cloud_cover if max_cloud_cover is None else max_cloud_cover
    )
    if not 0 <= threshold <= 100:
        raise BadRequestError("Порог облачности — от 0 до 100 %")
    geometry = await vineyard_geojson(session, vineyard)
    scenes = await copernicus.search(geometry, start, end, cloud_filter(threshold))
    return deduplicate(scenes)
