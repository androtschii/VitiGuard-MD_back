import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest

from app.api.deps import get_copernicus
from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations import copernicus
from app.integrations.copernicus import CopernicusClient

SETTINGS = Settings(_env_file=None)
POINT = {"type": "Point", "coordinates": [28.86, 47.14]}
START = datetime(2026, 6, 1, tzinfo=UTC)
END = datetime(2026, 10, 8, 23, 59, 59, tzinfo=UTC)


def feature(product_id: str, day: int, cloud: float | None = 5.0) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "datetime": f"2026-09-{day:02}T09:10:31.024000Z",
        "platform": "sentinel-2a",
        "grid:code": "MGRS-35TPN",
    }
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return {
        "id": product_id,
        "properties": properties,
        "assets": {
            name: {"href": f"s3://eodata/{product_id}/{name}.jp2"}
            for name in (
                "B04_10m",
                "B05_20m",
                "B08_10m",
                "B8A_20m",
                "SCL_20m",
                "AOT_10m",
            )
        },
    }


def client_for(
    pages: list[dict[str, Any]], seen: list[httpx.Request]
) -> CopernicusClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=pages[len(seen) - 1])

    return CopernicusClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), SETTINGS
    )


async def test_search_sends_stac_query_and_parses_scenes() -> None:
    seen: list[httpx.Request] = []
    client = client_for([{"features": [feature("S2A_X", 8, 31.98)], "links": []}], seen)

    scenes = await client.search(POINT, START, END)

    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://stac.dataspace.copernicus.eu/v1/search"
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["intersects"] == POINT
    assert body["datetime"] == "2026-06-01T00:00:00Z/2026-10-08T23:59:59Z"
    assert body["sortby"] == [{"field": "properties.datetime", "direction": "asc"}]
    assert "filter" not in body
    (scene,) = scenes
    assert scene.product_id == "S2A_X"
    assert scene.acquired_at == datetime(2026, 9, 8, 9, 10, 31, 24000, tzinfo=UTC)
    assert scene.cloud_cover == 31.98
    assert scene.tile == "35TPN"
    assert set(scene.assets) == {"B04", "B05", "B08", "B8A", "SCL"}
    assert scene.assets["B04"] == "s3://eodata/S2A_X/B04_10m.jp2"


async def test_pages_are_followed_by_next_link() -> None:
    next_body = {"collections": ["sentinel-2-l2a"], "token": "page-2"}
    pages = [
        {
            "features": [feature("A", 1)],
            "links": [
                {
                    "rel": "next",
                    "href": "https://stac.example/search",
                    "method": "POST",
                    "body": next_body,
                }
            ],
        },
        {"features": [feature("B", 2)], "links": []},
    ]
    seen: list[httpx.Request] = []

    scenes = await client_for(pages, seen).search(POINT, START, END)

    assert [s.product_id for s in scenes] == ["A", "B"]
    assert str(seen[1].url) == "https://stac.example/search"
    assert json.loads(seen[1].content) == next_body


async def test_extra_filter_is_sent_as_cql2() -> None:
    seen: list[httpx.Request] = []
    cql = {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10]}

    await client_for([{"features": []}], seen).search(POINT, START, END, cql)

    body = json.loads(seen[0].content)
    assert body["filter-lang"] == "cql2-json"
    assert body["filter"] == cql


async def test_missing_cloud_cover_counts_as_fully_cloudy() -> None:
    seen: list[httpx.Request] = []
    client = client_for([{"features": [feature("A", 1, cloud=None)]}], seen)

    (scene,) = await client.search(POINT, START, END)

    assert scene.cloud_cover == 100


async def test_runaway_search_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copernicus, "MAX_SCENES", 2)
    endless = {
        "features": [feature("A", 1), feature("B", 2)],
        "links": [
            {"rel": "next", "href": "https://stac.example/search", "method": "GET"}
        ],
    }
    seen: list[httpx.Request] = []

    with pytest.raises(ExternalServiceError, match="сузьте область или период"):
        await client_for([endless, endless, endless], seen).search(POINT, START, END)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="maintenance"),
        httpx.Response(200, text="<html/>"),
        httpx.Response(
            200, json={"features": [{"id": "A", "properties": {}, "assets": {}}]}
        ),
    ],
    ids=["server-error", "not-json", "broken-feature"],
)
async def test_bad_responses_become_external_service_error(
    response: httpx.Response,
) -> None:
    client = CopernicusClient(
        httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)), SETTINGS
    )

    with pytest.raises(
        ExternalServiceError, match="Каталог спутниковых снимков недоступен"
    ):
        await client.search(POINT, START, END)


async def test_dependency_uses_the_shared_http_client() -> None:
    async with httpx.AsyncClient() as http:
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(http_client=http))
        )

        client = get_copernicus(request, SETTINGS)  # type: ignore[arg-type]

        assert client._http is http
