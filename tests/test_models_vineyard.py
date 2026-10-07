import pytest
from geoalchemy2 import Geography, WKTElement
from sqlalchemy import cast, delete, func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User, Vineyard

pytestmark = pytest.mark.integration

# Прямоугольный участок около Крикова, примерно 1,7 га
PLOT = (
    "MULTIPOLYGON(((28.860 47.140, 28.862 47.140, 28.862 47.141,"
    " 28.860 47.141, 28.860 47.140)))"
)


async def add_vineyard(session: AsyncSession, wkt: str = PLOT) -> Vineyard:
    owner = User(email="owner@example.md", hashed_password="hash")
    vineyard = Vineyard(
        owner=owner, name="Крикова, участок 1", geom=WKTElement(wkt, srid=4326)
    )
    session.add(vineyard)
    await session.flush()
    return vineyard


async def test_geometry_type_and_srid(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)

    row = (
        await session.execute(
            select(func.GeometryType(Vineyard.geom), func.ST_SRID(Vineyard.geom)).where(
                Vineyard.id == vineyard.id
            )
        )
    ).one()

    assert tuple(row) == ("MULTIPOLYGON", 4326)


async def test_area_on_ellipsoid(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)

    area_ha = await session.scalar(
        select(func.ST_Area(cast(Vineyard.geom, Geography)) / 10_000).where(
            Vineyard.id == vineyard.id
        )
    )

    assert area_ha is not None
    assert 1.6 < area_ha < 1.8


async def test_point_is_rejected(session: AsyncSession) -> None:
    with pytest.raises(DBAPIError, match="does not match column type"):
        await add_vineyard(session, "POINT(28.86 47.14)")


async def test_spatial_index_exists(session: AsyncSession) -> None:
    indexdef = await session.scalar(
        text(
            "SELECT indexdef FROM pg_indexes"
            " WHERE tablename = 'vineyards' AND indexname = 'idx_vineyards_geom'"
        )
    )

    assert indexdef is not None
    assert "USING gist" in indexdef


async def test_vineyards_deleted_with_owner(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)

    await session.execute(delete(User).where(User.id == vineyard.owner_id))

    count = await session.scalar(select(func.count()).select_from(Vineyard))
    assert count == 0
