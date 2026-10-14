from typing import Any

import httpx2 as httpx
import pytest

from app.core.config import Settings
from app.core.errors import ExternalServiceError
from app.integrations.overpass import OverpassClient


def nodes(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"lon": lon, "lat": lat} for lon, lat in points]


SQUARE = [(28.86, 47.14), (28.862, 47.14), (28.862, 47.141), (28.86, 47.14)]
HOLE = [(28.8605, 47.1402), (28.861, 47.1402), (28.861, 47.1404), (28.8605, 47.1402)]
OTHER = [(x + 0.01, y) for x, y in SQUARE]

ELEMENTS: list[dict[str, Any]] = [
    {"type": "way", "id": 1, "tags": {"name": "Cricova"}, "geometry": nodes(SQUARE)},
    {"type": "way", "id": 2, "geometry": nodes(SQUARE[:-1])},
    {
        "type": "relation",
        "id": 3,
        "members": [
            {"type": "way", "role": "outer", "geometry": nodes(SQUARE)},
            {"type": "way", "role": "inner", "geometry": nodes(HOLE)},
            {"type": "way", "role": "outer", "geometry": nodes(OTHER)},
            {"type": "node", "role": "label"},
        ],
    },
    {
        "type": "relation",
        "id": 4,
        "members": [{"type": "way", "role": "outer", "geometry": nodes(SQUARE[:2])}],
    },
    {"type": "relation", "id": 5, "members": [{"type": "node", "role": "label"}]},
]


def client_with(handler: Any) -> OverpassClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OverpassClient(http, Settings(_env_file=None, environment="test"))


async def test_parses_ways_and_multipolygons() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode())
        return httpx.Response(200, json={"elements": ELEMENTS})

    result = await client_with(handler).vineyards()

    assert "landuse" in sent[0]
    assert "ISO3166-1" in sent[0]
    assert [v.osm_id for v in result.vineyards] == ["way/1", "relation/3"]
    assert result.vineyards[0].name == "Cricova"
    assert result.vineyards[0].geometry == {
        "type": "MultiPolygon",
        "coordinates": [[[list(p) for p in SQUARE]]],
    }
    # Вырез — у первой внешней части, вторая внешняя — отдельная часть
    assert result.vineyards[1].geometry["coordinates"] == [
        [[list(p) for p in SQUARE], [list(p) for p in HOLE]],
        [[list(p) for p in OTHER]],
    ]
    # Незамкнутый путь, мультиполигон из кусков и пустой мультиполигон
    assert result.skipped == 3


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(504, text="Gateway Timeout"),
        httpx.Response(200, text="<html>busy</html>"),
        httpx.Response(200, json={"remark": "runtime error"}),
    ],
)
async def test_failures_become_external_service_error(
    response: httpx.Response,
) -> None:
    with pytest.raises(ExternalServiceError, match="Overpass"):
        await client_with(lambda _: response).vineyards()


async def test_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет сети", request=request)

    with pytest.raises(ExternalServiceError):
        await client_with(handler).vineyards()
