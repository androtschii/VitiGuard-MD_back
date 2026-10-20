"""Сводная статистика по районам Молдовы и определение района участка."""

import uuid
from typing import Any

from geoalchemy2 import Geography
from geoalchemy2 import functions as geo
from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.cache import Cache, cached
from app.models.osm import OsmVineyard
from app.models.region import District
from app.models.vineyard import Vineyard
from app.schemas.region import DistrictRef, DistrictStats


def _area_ha(geom: Any) -> Any:
    return geo.ST_Area(cast(geom, Geography)) / 10_000


def _per_district(model: type[OsmVineyard] | type[Vineyard]) -> Any:
    # Полигон относится к району, в котором лежит его внутренняя точка: виноградник
    # на границе районов не считается дважды, а поиск идёт по пространственному индексу
    return (
        select(
            District.iso.label("iso"),
            func.count().label("count"),
            func.sum(_area_ha(model.geom)).label("area_ha"),
        )
        .join(
            model,
            geo.ST_Intersects(District.geom, geo.ST_PointOnSurface(model.geom)),
        )
        .group_by(District.iso)
    )


async def _osm_districts(session: AsyncSession) -> list[list[Any]]:
    """Общая для всех часть статистики: площадь районов и виноградники OSM.
    Это площади 17 тысяч полигонов — самая тяжёлая часть запроса."""
    osm = _per_district(OsmVineyard).subquery()
    rows = await session.execute(
        select(
            District.iso,
            District.name,
            _area_ha(District.geom),
            func.coalesce(osm.c.count, 0),
            func.coalesce(osm.c.area_ha, 0),
        )
        .outerjoin(osm, osm.c.iso == District.iso)
        .order_by(District.name)
    )
    return [
        [iso, name, float(area), int(count), float(osm_ha)]
        for iso, name, area, count, osm_ha in rows
    ]


async def district_stats(
    session: AsyncSession, owner_id: uuid.UUID, cache: Cache
) -> list[DistrictStats]:
    """Все районы по алфавиту: площадь, виноградники OSM и участки пользователя.
    Участки других пользователей не учитываются — они видны только владельцу.
    Часть OSM общая для всех и берётся из кэша; участки пользователя считаются
    каждый раз — их единицы, и они меняются чаще."""
    districts = await cached(cache, "gis:districts", lambda: _osm_districts(session))
    own = {
        iso: (int(count), float(area))
        for iso, count, area in await session.execute(
            _per_district(Vineyard).where(Vineyard.owner_id == owner_id)
        )
    }
    return [
        DistrictStats(
            iso=iso,
            name=name,
            area_ha=round(district_ha, 1),
            osm_vineyards=osm_count,
            osm_vineyard_area_ha=round(osm_ha, 1),
            osm_vineyard_share=round(min(osm_ha / district_ha, 1.0), 4),
            my_vineyards=own.get(iso, (0, 0.0))[0],
            my_vineyard_area_ha=round(own.get(iso, (0, 0.0))[1], 2),
        )
        for iso, name, district_ha, osm_count, osm_ha in districts
    ]


async def district_of(session: AsyncSession, vineyard: Vineyard) -> DistrictRef | None:
    """Район участка; None — участок за пределами Молдовы."""
    own = select(Vineyard.geom).where(Vineyard.id == vineyard.id).scalar_subquery()
    row = (
        await session.execute(
            select(District.iso, District.name).where(
                geo.ST_Intersects(District.geom, geo.ST_PointOnSurface(own))
            )
        )
    ).first()
    return None if row is None else DistrictRef(iso=row.iso, name=row.name)
