import json
from datetime import UTC, datetime
from typing import Any

import httpx2 as httpx
import pytest
from geoalchemy2 import WKTElement
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import BadRequestError, NotFoundError
from app.integrations.copernicus import CopernicusClient, Scene
from app.models import User, Vineyard
from app.services.satellite import deduplicate, find_scenes, vineyard_geojson

PLOT = (
    "MULTIPOLYGON(((28.860 47.140, 28.862 47.140, 28.862 47.141,"
    " 28.860 47.141, 28.860 47.140)))"
)
START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)


def scene(
    product_id: str,
    day: int,
    cloud: float,
    tile: str = "35TPN",
    platform: str = "sentinel-2a",
) -> Scene:
    return Scene(
        product_id=product_id,
        acquired_at=datetime(2026, 9, day, 9, 10, 31, 24000, tzinfo=UTC),
        cloud_cover=cloud,
        platform=platform,
        tile=tile,
        assets={},
    )


def test_reprocessed_scene_keeps_newest_baseline() -> None:
    old = scene("S2A_MSIL2A_20260908T091031_N0511_R050_T35TPN_X", 8, 3.0)
    new = scene("S2A_MSIL2A_20260908T091031_N0513_R050_T35TPN_Y", 8, 5.0)

    assert deduplicate([new, old]) == [new]


def test_scene_from_two_tiles_keeps_less_cloudy() -> None:
    tile_a = scene("S2A_MSIL2A_20260908T091031_N0513_R050_T35TPN_X", 8, 20.0)
    tile_b = scene(
        "S2A_MSIL2A_20260908T091031_N0513_R050_T35TNN_Y", 8, 4.0, tile="35TNN"
    )

    assert deduplicate([tile_a, tile_b]) == [tile_b]


def test_different_acquisitions_are_kept_in_date_order() -> None:
    later = scene("S2A_MSIL2A_20260918T091031_N0513_X", 18, 1.0)
    earlier = scene("S2A_MSIL2A_20260908T091031_N0513_Y", 8, 1.0)
    other_satellite = scene(
        "S2B_MSIL2A_20260908T091031_N0513_Z", 8, 1.0, platform="sentinel-2b"
    )

    result = deduplicate([later, earlier, other_satellite])

    assert [s.product_id[:3] for s in result] == ["S2A", "S2B", "S2A"]
    assert result[-1] is later


def feature(product_id: str, day: int) -> dict[str, Any]:
    return {
        "id": product_id,
        "properties": {
            "datetime": f"2026-09-{day:02}T09:10:31.024Z",
            "platform": "sentinel-2a",
            "eo:cloud_cover": 2.0,
            "grid:code": "MGRS-35TPN",
        },
        "assets": {},
    }


async def add_vineyard(session: AsyncSession) -> Vineyard:
    vineyard = Vineyard(
        owner=User(email="owner@example.md", hashed_password="hash"),
        name="Крикова",
        geom=WKTElement(PLOT, srid=4326),
    )
    session.add(vineyard)
    await session.flush()
    return vineyard


@pytest.mark.integration
async def test_scenes_are_searched_by_vineyard_polygon(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("S2A_MSIL2A_20260908T091031_N0511_R050_T35TPN_A", 8),
                    feature("S2A_MSIL2A_20260908T091031_N0513_R050_T35TPN_B", 8),
                ]
            },
        )

    client = CopernicusClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        Settings(_env_file=None),
    )

    scenes = await find_scenes(session, client, vineyard, START, END)

    assert sent[0]["intersects"]["type"] == "MultiPolygon"
    assert sent[0]["intersects"]["coordinates"][0][0][0] == [28.86, 47.14]
    assert [s.product_id for s in scenes] == [
        "S2A_MSIL2A_20260908T091031_N0513_R050_T35TPN_B"
    ]


@pytest.mark.integration
async def test_period_must_be_ordered(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    client = CopernicusClient(
        httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("запроса быть не должно")
            )
        ),
        Settings(_env_file=None),
    )

    with pytest.raises(BadRequestError):
        await find_scenes(session, client, vineyard, END, START)


@pytest.mark.integration
async def test_deleted_vineyard_is_not_found(session: AsyncSession) -> None:
    vineyard = await add_vineyard(session)
    await session.execute(text("DELETE FROM vineyards"))

    with pytest.raises(NotFoundError):
        await vineyard_geojson(session, vineyard)
