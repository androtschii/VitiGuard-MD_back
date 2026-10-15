"""Сводная статистика по районам Молдовы и определение района участка."""

import uuid
from typing import Any

from geoalchemy2 import Geography
from geoalchemy2 import functions as geo
from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

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


async def district_stats(
    session: AsyncSession, owner_id: uuid.UUID
) -> list[DistrictStats]:
    """Все районы по алфавиту: площадь, виноградники OSM и участки пользователя.
    Участки других пользователей не учитываются — они видны только владельцу."""
    osm = _per_district(OsmVineyard).subquery()
    own = _per_district(Vineyard).where(Vineyard.owner_id == owner_id).subquery()
    area = _area_ha(District.geom)
    rows = await session.execute(
        select(
            District.iso,
            District.name,
            area,
            func.coalesce(osm.c.count, 0),
            func.coalesce(osm.c.area_ha, 0),
            func.coalesce(own.c.count, 0),
            func.coalesce(own.c.area_ha, 0),
        )
        .outerjoin(osm, osm.c.iso == District.iso)
        .outerjoin(own, own.c.iso == District.iso)
        .order_by(District.name)
    )
    return [
        DistrictStats(
            iso=iso,
            name=name,
            area_ha=round(float(district_ha), 1),
            osm_vineyards=int(osm_count),
            osm_vineyard_area_ha=round(float(osm_ha), 1),
            osm_vineyard_share=round(min(float(osm_ha) / float(district_ha), 1.0), 4),
            my_vineyards=int(own_count),
            my_vineyard_area_ha=round(float(own_ha), 2),
        )
        for iso, name, district_ha, osm_count, osm_ha, own_count, own_ha in rows
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
