from datetime import UTC, datetime

import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    IndexType,
    ProcessingStatus,
    SatelliteScan,
    User,
    VegetationIndex,
    Vineyard,
)

pytestmark = pytest.mark.integration

PLOT = "MULTIPOLYGON(((28.86 47.14, 28.87 47.14, 28.87 47.15, 28.86 47.14)))"
PRODUCT = "S2B_MSIL2A_20260715T090559_N0511_R050_T35TMN_20260715T112233"


def make_vineyard() -> Vineyard:
    owner = User(email="grower@example.md", hashed_password="hash")
    return Vineyard(owner=owner, name="Участок", geom=WKTElement(PLOT, srid=4326))


def make_scan(vineyard: Vineyard, **kwargs: object) -> SatelliteScan:
    values: dict[str, object] = {
        "product_id": PRODUCT,
        "acquired_at": datetime(2026, 7, 15, 9, 5, tzinfo=UTC),
        "cloud_cover": 12.5,
    }
    values.update(kwargs)
    return SatelliteScan(vineyard=vineyard, **values)


def make_index(scan: SatelliteScan, **kwargs: object) -> VegetationIndex:
    values: dict[str, object] = {
        "index_type": IndexType.NDVI,
        "mean_value": 0.62,
        "min_value": 0.31,
        "max_value": 0.81,
        "std_value": 0.08,
    }
    values.update(kwargs)
    return VegetationIndex(scan=scan, **values)


async def test_scan_with_indices(session: AsyncSession) -> None:
    scan = make_scan(make_vineyard())
    make_index(scan)
    make_index(
        scan,
        index_type=IndexType.NDRE,
        mean_value=0.28,
        min_value=0.12,
        max_value=0.45,
    )
    session.add(scan)
    await session.flush()
    await session.refresh(scan)

    types = await session.scalars(
        select(VegetationIndex.index_type)
        .where(VegetationIndex.scan_id == scan.id)
        .order_by(VegetationIndex.index_type)
    )

    assert scan.status is ProcessingStatus.PENDING
    assert list(types) == [IndexType.NDRE, IndexType.NDVI]


async def test_same_product_rejected_for_vineyard(session: AsyncSession) -> None:
    vineyard = make_vineyard()
    session.add_all([make_scan(vineyard), make_scan(vineyard)])

    with pytest.raises(
        IntegrityError, match="uq_satellite_scans_vineyard_id_product_id"
    ):
        await session.flush()


async def test_one_value_per_index_type(session: AsyncSession) -> None:
    scan = make_scan(make_vineyard())
    make_index(scan)
    make_index(scan)
    session.add(scan)

    with pytest.raises(
        IntegrityError, match="uq_vegetation_indices_scan_id_index_type"
    ):
        await session.flush()


@pytest.mark.parametrize("cloud_cover", [-1, 100.5])
async def test_cloud_cover_range(session: AsyncSession, cloud_cover: float) -> None:
    session.add(make_scan(make_vineyard(), cloud_cover=cloud_cover))

    with pytest.raises(IntegrityError, match="ck_satellite_scans_cloud_cover_range"):
        await session.flush()


@pytest.mark.parametrize(
    ("fields", "constraint"),
    [
        ({"max_value": 1.2}, "ck_vegetation_indices_values_range"),
        ({"mean_value": 0.9}, "ck_vegetation_indices_values_range"),
        ({"std_value": -0.1}, "ck_vegetation_indices_std_non_negative"),
    ],
)
async def test_index_value_constraints(
    session: AsyncSession, fields: dict[str, object], constraint: str
) -> None:
    scan = make_scan(make_vineyard())
    make_index(scan, **fields)
    session.add(scan)

    with pytest.raises(IntegrityError, match=constraint):
        await session.flush()


async def test_scans_deleted_with_vineyard(session: AsyncSession) -> None:
    scan = make_scan(make_vineyard())
    make_index(scan)
    session.add(scan)
    await session.flush()

    await session.execute(delete(Vineyard).where(Vineyard.id == scan.vineyard_id))

    assert await session.scalar(select(func.count()).select_from(SatelliteScan)) == 0
    assert await session.scalar(select(func.count()).select_from(VegetationIndex)) == 0
