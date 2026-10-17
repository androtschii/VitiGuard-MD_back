import json
from datetime import UTC, datetime
from typing import Any

import httpx2 as httpx

from app.core.config import Settings
from app.integrations.earth_search import (
    EarthScene,
    EarthSearchClient,
    cloud_query,
    deduplicate,
)

SETTINGS = Settings(_env_file=None)
POINT = {"type": "Point", "coordinates": [28.86, 47.14]}
COG = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"


def feature(
    product_id: str = "S2A_35TPN_20260829_1_L2A",
    day: int = 29,
    cloud: float = 5.0,
    offset_applied: bool | None = True,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "datetime": f"2026-08-{day:02}T09:17:25.539000Z",
        "platform": "sentinel-2a",
        "eo:cloud_cover": cloud,
        "grid:code": "MGRS-35TPN",
    }
    if offset_applied is not None:
        properties["earthsearch:boa_offset_applied"] = offset_applied
    reflectance = [{"nodata": 0, "scale": 0.0001, "offset": -0.1}]
    return {
        "id": product_id,
        "properties": properties,
        "assets": {
            "red": {"href": f"{COG}/B04.tif", "raster:bands": reflectance},
            "rededge1": {"href": f"{COG}/B05.tif", "raster:bands": reflectance},
            "nir": {"href": f"{COG}/B08.tif", "raster:bands": reflectance},
            "nir08": {"href": f"{COG}/B8A.tif", "raster:bands": reflectance},
            "scl": {"href": f"{COG}/SCL.tif", "raster:bands": [{"nodata": 0}]},
            "visual": {"href": f"{COG}/TCI.tif"},
        },
    }


async def search(features: list[dict[str, Any]]) -> tuple[list[EarthScene], Any]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"features": features, "links": []})

    client = EarthSearchClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), SETTINGS
    )
    scenes = await client.search(
        POINT,
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 31, tzinfo=UTC),
        extra_body=cloud_query(40),
    )
    return scenes, seen[0]


async def test_search_uses_query_extension_and_parses_bands() -> None:
    scenes, request = await search([feature()])

    body = json.loads(request.content)
    assert str(request.url) == f"{SETTINGS.earth_search_url}/search"
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["query"] == {"eo:cloud_cover": {"lte": 40}}
    assert "filter" not in body
    [scene] = scenes
    assert scene.product_id == "S2A_35TPN_20260829_1_L2A"
    assert scene.tile == "35TPN"
    assert scene.acquired_at.tzinfo is not None
    assert set(scene.bands) == {"B04", "B05", "B08", "B8A", "SCL"}
    red = scene.bands["B04"]
    assert red.href.endswith("B04.tif")
    # Earth Search уже вычел смещение — второй раз оно не применяется
    assert (red.scale, red.offset) == (0.0001, 0.0)
    assert (scene.bands["SCL"].scale, scene.bands["SCL"].offset) == (1.0, 0.0)


async def test_offset_from_metadata_when_not_applied() -> None:
    scenes, _ = await search([feature(offset_applied=None)])

    assert scenes[0].bands["B08"].offset == -0.1


def scene(product_id: str, hour: int, cloud: float, tile: str) -> EarthScene:
    return EarthScene(
        product_id=product_id,
        acquired_at=datetime(2026, 6, 10, hour, 5, tzinfo=UTC),
        cloud_cover=cloud,
        platform="sentinel-2a",
        tile=tile,
        bands={},
    )


def test_deduplicate_keeps_one_scene_per_satellite_and_day() -> None:
    old = scene("S2A_35TPN_20260610_0_L2A", 9, 1.0, "35TPN")
    newer = scene("S2A_35TPN_20260610_1_L2A", 9, 5.0, "35TPN")
    other_tile = scene("S2A_35TNN_20260610_1_L2A", 9, 2.0, "35TNN")
    next_day = scene("S2A_35TPN_20260611_0_L2A", 9, 3.0, "35TPN").model_copy(
        update={"acquired_at": datetime(2026, 6, 11, 9, 5, tzinfo=UTC)}
    )
    odd = scene("weird-id", 9, 0.0, "35TPN").model_copy(
        update={"platform": "sentinel-2b"}
    )

    result = deduplicate([newer, next_day, old, odd, other_tile])

    # Новейшая обработка, затем меньшая облачность; другой спутник — отдельно
    assert [s.product_id for s in result] == [
        "S2A_35TNN_20260610_1_L2A",
        "weird-id",
        "S2A_35TPN_20260611_0_L2A",
    ]
