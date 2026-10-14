"""Справочный слой виноградников OpenStreetMap: импорт, выборка для карты и
контекст местности участка пользователя."""

import json
from dataclasses import dataclass
from typing import Any

from geoalchemy2 import Geography
from geoalchemy2 import functions as geo
from sqlalchemy import CursorResult, cast, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.overpass import OsmVineyard as OsmRecord
from app.models.osm import OsmVineyard
from app.models.vineyard import Vineyard

# Пачки по 500: один INSERT на все 17 тысяч полигонов был бы огромным запросом
BATCH = 500
# Слой для карты: больше полигонов в одном ответе браузер рисует медленно
MAX_FEATURES = 2000
NEARBY_M = 1000


@dataclass(frozen=True)
class ImportStats:
    imported: int
    deleted: int


def _valid_geom(geometry: dict[str, Any]) -> Any:
    # Контуры OSM рисуют люди: самопересечения встречаются. ST_MakeValid чинит
    # их, ST_CollectionExtract оставляет только полигоны, ST_Multi — тип столбца
    return geo.ST_Multi(
        geo.ST_CollectionExtract(
            geo.ST_MakeValid(geo.ST_GeomFromGeoJSON(json.dumps(geometry))), 3
        )
    )


async def import_osm_vineyards(
    session: AsyncSession, records: list[OsmRecord]
) -> ImportStats:
    """Полная синхронизация со свежей выгрузкой: новые полигоны добавляются,
    изменённые обновляются, исчезнувшие из OSM удаляются."""
    for start in range(0, len(records), BATCH):
        rows = [
            {"osm_id": r.osm_id, "name": r.name, "geom": _valid_geom(r.geometry)}
            for r in records[start : start + BATCH]
        ]
        statement = insert(OsmVineyard).values(rows)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[OsmVineyard.osm_id],
                set_={
                    "name": statement.excluded.name,
                    "geom": statement.excluded.geom,
                    "updated_at": func.now(),
                },
            )
        )
    stale: CursorResult[Any] = await session.execute(  # type: ignore[assignment]
        delete(OsmVineyard).where(
            OsmVineyard.osm_id.not_in([r.osm_id for r in records])
        )
    )
    return ImportStats(imported=len(records), deleted=stale.rowcount)


async def osm_vineyards_in_bbox(
    session: AsyncSession, west: float, south: float, east: float, north: float
) -> dict[str, Any]:
    envelope = geo.ST_MakeEnvelope(west, south, east, north, 4326)
    area_ha = geo.ST_Area(cast(OsmVineyard.geom, Geography)) / 10_000
    rows = await session.execute(
        select(
            OsmVineyard.osm_id,
            OsmVineyard.name,
            area_ha,
            geo.ST_AsGeoJSON(OsmVineyard.geom),
        )
        .where(geo.ST_Intersects(OsmVineyard.geom, envelope))
        .order_by(OsmVineyard.osm_id)
        .limit(MAX_FEATURES)
    )
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "osm_id": osm_id,
                    "name": name,
                    "area_ha": round(float(area), 4),
                },
                "geometry": json.loads(geometry),
            }
            for osm_id, name, area, geometry in rows
        ],
    }


@dataclass(frozen=True)
class VineyardContext:
    # Доля площади участка, которая и в OSM размечена как виноградник
    osm_overlap_ratio: float
    osm_vineyards_within_1km: int
    nearest_osm_distance_m: float | None


async def vineyard_context(
    session: AsyncSession, vineyard: Vineyard
) -> VineyardContext:
    """Что вокруг участка по данным OSM. Совпадение с размеченным виноградником
    подтверждает контур; соседние виноградники — источник заражения: споры
    милдью и оидиума переносятся ветром на сотни метров."""
    own = select(Vineyard.geom).where(Vineyard.id == vineyard.id).scalar_subquery()
    own_geo = cast(own, Geography)
    osm_geo = cast(OsmVineyard.geom, Geography)

    covered = await session.scalar(
        select(
            geo.ST_Area(
                cast(
                    geo.ST_Intersection(own, geo.ST_Union(OsmVineyard.geom)), Geography
                )
            )
            / geo.ST_Area(own_geo)
        ).where(geo.ST_Intersects(OsmVineyard.geom, own))
    )
    nearby = await session.scalar(
        select(func.count())
        .select_from(OsmVineyard)
        .where(geo.ST_DWithin(osm_geo, own_geo, NEARBY_M))
    )
    nearest = await session.scalar(
        select(func.min(geo.ST_Distance(osm_geo, own_geo))).where(
            geo.ST_DWithin(osm_geo, own_geo, 50_000)
        )
    )
    return VineyardContext(
        osm_overlap_ratio=round(min(float(covered or 0), 1.0), 4),
        osm_vineyards_within_1km=int(nearby or 0),
        nearest_osm_distance_m=None if nearest is None else round(float(nearest), 1),
    )
