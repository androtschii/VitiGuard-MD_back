"""ГИС-запросы на уровне репозиториев и сервисов: GeoAlchemy2 и PostGIS в
асинхронной сессии, без HTTP. Каждый тест — в своей откатываемой транзакции."""

from typing import Any

import pytest
from geoalchemy2 import functions as geo
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.overpass import OsmVineyard as OsmRecord
from app.models import User
from app.models.osm import OsmVineyard
from app.models.vineyard import Vineyard
from app.repositories.vineyard import VineyardRepository
from app.services.osm import (
    import_osm_vineyards,
    osm_vineyards_in_bbox,
    vineyard_context,
)
from app.services.region import district_of

pytestmark = pytest.mark.integration

# Сотая градуса долготы на широте Крикова — около 758 м
LON_STEP_M = 757.7


def square(west: float, south: float = 47.14, side: float = 0.002) -> dict[str, Any]:
    ring = [
        [west, south],
        [west + side, south],
        [west + side, south + side / 2],
        [west, south + side / 2],
        [west, south],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


async def add_owner(session: AsyncSession, email: str = "grower@example.md") -> User:
    owner = User(email=email, hashed_password="hash")
    session.add(owner)
    await session.flush()
    return owner


async def add_vineyard(
    session: AsyncSession, owner: User, name: str, geometry: dict[str, Any]
) -> Vineyard:
    return await VineyardRepository(session).create_from_geojson(
        owner.id, name, geometry
    )


async def test_nearest_measures_meters_on_ellipsoid(session: AsyncSession) -> None:
    owner = await add_owner(session)
    neighbour = await add_owner(session, "neighbour@example.md")
    here = await add_vineyard(session, owner, "Здесь", square(28.860))
    east = await add_vineyard(session, owner, "Восток", square(28.870))
    await add_vineyard(session, owner, "Далеко", square(28.960))
    await add_vineyard(session, neighbour, "Чужой", square(28.864))

    found = await VineyardRepository(session).nearest(
        owner.id, 28.861, 47.1405, limit=5, max_distance_m=2_000
    )

    assert [view.vineyard.id for view, _ in found] == [here.id, east.id]
    # Точка внутри участка — расстояние 0, до соседнего — 0,009° долготы
    assert found[0][1] == 0
    assert found[1][1] == pytest.approx(0.9 * LON_STEP_M, rel=0.01)


async def test_view_has_geojson_and_area_of_all_parts(session: AsyncSession) -> None:
    owner = await add_owner(session)
    two_parts = {
        "type": "MultiPolygon",
        "coordinates": [
            square(28.860)["coordinates"],
            square(28.870)["coordinates"],
        ],
    }
    vineyard = await add_vineyard(session, owner, "Два куска", two_parts)

    view = await VineyardRepository(session).view(vineyard)

    assert view.geometry["type"] == "MultiPolygon"
    assert len(view.geometry["coordinates"]) == 2
    # 0,002° × 0,001° на широте 47° — около 1,7 га, два куска — вдвое больше
    assert view.area_ha == pytest.approx(3.4, rel=0.03)


async def test_owner_views_are_sorted_by_name(session: AsyncSession) -> None:
    owner = await add_owner(session)
    for index, name in enumerate(["Б", "А", "В"]):
        await add_vineyard(session, owner, name, square(28.860 + index * 0.01))

    views = await VineyardRepository(session).all_for_owner(owner.id)

    assert [view.vineyard.name for view in views] == ["А", "Б", "В"]


def bowtie(osm_id: str) -> OsmRecord:
    # Восьмёрка: контур пересекает сам себя, как бывает в ручной разметке OSM
    ring = [
        [28.860, 47.140],
        [28.862, 47.141],
        [28.862, 47.140],
        [28.860, 47.141],
        [28.860, 47.140],
    ]
    return OsmRecord(
        osm_id=osm_id, name=None, geometry={"type": "Polygon", "coordinates": [ring]}
    )


async def test_osm_import_repairs_updates_and_removes(session: AsyncSession) -> None:
    await import_osm_vineyards(
        session,
        [
            bowtie("way/1"),
            OsmRecord(osm_id="way/2", name="Старое", geometry=square(28.870)),
        ],
    )

    stats = await import_osm_vineyards(
        session,
        [bowtie("way/1"), OsmRecord(osm_id="way/3", name=None, geometry=square(28.9))],
    )

    assert (stats.imported, stats.deleted) == (2, 1)
    rows = (
        await session.execute(
            select(
                OsmVineyard.osm_id,
                geo.ST_IsValid(OsmVineyard.geom),
                func.GeometryType(OsmVineyard.geom),
            ).order_by(OsmVineyard.osm_id)
        )
    ).all()
    assert [tuple(row) for row in rows] == [
        ("way/1", True, "MULTIPOLYGON"),
        ("way/3", True, "MULTIPOLYGON"),
    ]


async def test_osm_layer_takes_polygons_touching_window(session: AsyncSession) -> None:
    await import_osm_vineyards(
        session,
        [
            OsmRecord(osm_id="way/1", name="Внутри", geometry=square(28.860)),
            OsmRecord(osm_id="way/2", name="На краю", geometry=square(28.8695)),
            OsmRecord(osm_id="way/3", name="Снаружи", geometry=square(28.900)),
        ],
    )

    layer = await osm_vineyards_in_bbox(session, 28.85, 47.13, 28.87, 47.15)

    features = layer["features"]
    assert [f["properties"]["osm_id"] for f in features] == ["way/1", "way/2"]
    assert features[0]["properties"]["area_ha"] == pytest.approx(1.7, rel=0.03)
    assert features[0]["geometry"]["type"] == "MultiPolygon"


async def test_context_overlap_and_neighbours(session: AsyncSession) -> None:
    owner = await add_owner(session)
    vineyard = await add_vineyard(session, owner, "Участок", square(28.860))
    await import_osm_vineyards(
        session,
        [
            # Восточная половина участка размечена в OSM как виноградник
            OsmRecord(osm_id="way/1", name=None, geometry=square(28.861)),
            OsmRecord(osm_id="way/2", name=None, geometry=square(28.866)),
            OsmRecord(osm_id="way/3", name=None, geometry=square(28.900)),
        ],
    )

    context = await vineyard_context(session, vineyard)

    assert context.osm_overlap_ratio == pytest.approx(0.5, abs=0.01)
    assert context.osm_vineyards_within_1km == 2
    assert context.nearest_osm_distance_m == 0


async def test_context_without_osm_around(session: AsyncSession) -> None:
    owner = await add_owner(session)
    vineyard = await add_vineyard(session, owner, "Участок", square(28.860))

    context = await vineyard_context(session, vineyard)

    assert context.osm_overlap_ratio == 0
    assert context.osm_vineyards_within_1km == 0
    assert context.nearest_osm_distance_m is None


async def test_district_by_point_on_surface(session: AsyncSession) -> None:
    owner = await add_owner(session)
    cricova = await add_vineyard(session, owner, "Крикова", square(28.860))
    sea = await add_vineyard(session, owner, "Море", square(30.860, 44.64))

    district = await district_of(session, cricova)

    assert district is not None
    assert (district.iso, district.name) == ("MD-CU", "Chișinău")
    assert await district_of(session, sea) is None
