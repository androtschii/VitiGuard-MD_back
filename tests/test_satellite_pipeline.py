"""Спутниковый модуль целиком на подменённых данных: настоящий клиент Earth
Search получает ответ каталога из подмены HTTP (две страницы, повтор снимка в
соседнем тайле, переобработанная версия, облачный снимок), каналы читаются из
локальных GeoTIFF, а задача обработки сохраняет индексы и картограммы, которые
потом видны в истории участка и на тайлах карты."""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient
from raster_scene import CLOUD, PARCEL, make_scene

from app.core.config import Settings
from app.integrations.earth_search import EarthScene
from app.tasks import satellite

pytestmark = pytest.mark.integration

SEARCH = "https://earth-search.example/v1/search"


def feature(scene: EarthScene, offset_applied: bool = True) -> dict[str, Any]:
    """Снимок в том виде, в каком его отдаёт Earth Search (STAC)."""
    names = {"B04": "red", "B05": "rededge1", "B08": "nir", "B8A": "nir08"}
    reflectance = [{"nodata": 0, "scale": 0.0001, "offset": scene.bands["B04"].offset}]
    assets: dict[str, Any] = {
        name: {"href": scene.bands[band].href, "raster:bands": reflectance}
        for band, name in names.items()
    }
    assets["scl"] = {"href": scene.bands["SCL"].href, "raster:bands": [{"nodata": 0}]}
    return {
        "id": scene.product_id,
        "properties": {
            "datetime": scene.acquired_at.isoformat().replace("+00:00", "Z"),
            "platform": scene.platform,
            "eo:cloud_cover": scene.cloud_cover,
            "grid:code": f"MGRS-{scene.tile}",
            "earthsearch:boa_offset_applied": offset_applied,
        },
        "assets": assets,
    }


def at(day: int, month: int = 7) -> datetime:
    return datetime(2026, month, day, 9, 17, tzinfo=UTC)


@pytest.fixture
def catalog(tmp_path: Path) -> list[dict[str, Any]]:
    """Страницы ответа каталога за сезон."""
    first = make_scene(
        tmp_path, product_id="S2A_35TPN_20260710_0_L2A", acquired_at=at(10)
    )
    # Тот же пролёт, переобработанный новой версией 04.00: к значениям прибавлена
    # 1000, а смещение −0,1 Earth Search не вычел — его применяет сервер
    reprocessed = make_scene(
        tmp_path,
        product_id="S2A_35TPN_20260710_1_L2A",
        acquired_at=at(10),
        offset=-0.1,
        red=1500,
    )
    neighbour_tile = make_scene(
        tmp_path, product_id="S2A_35TPM_20260710_0_L2A", acquired_at=at(10)
    )
    neighbour_tile = neighbour_tile.model_copy(
        update={"tile": "35TPM", "cloud_cover": 30}
    )
    cloudy = make_scene(
        tmp_path, scl=CLOUD, product_id="S2B_35TPN_20260725_0_L2A", acquired_at=at(25)
    )
    cloudy = cloudy.model_copy(update={"platform": "sentinel-2b"})
    august = make_scene(
        tmp_path, product_id="S2A_35TPN_20260809_0_L2A", acquired_at=at(9, 8)
    )
    return [
        {
            "features": [feature(first), feature(neighbour_tile)],
            "links": [
                {
                    "rel": "next",
                    "href": SEARCH,
                    "method": "POST",
                    "body": {"next": "page-2"},
                }
            ],
        },
        {
            "features": [
                feature(reprocessed, offset_applied=False),
                feature(cloudy),
                feature(august),
            ],
            "links": [],
        },
    ]


def login(client: TestClient, email: str = "grower@example.md") -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def tile_of(lon: float, lat: float, z: int) -> str:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return f"{z}/{x}/{y}"


def test_season_from_catalog_to_history_and_tiles(
    api_client: TestClient,
    catalog: list[dict[str, Any]],
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = login(api_client)
    vineyard = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": PARCEL},
    ).json()["id"]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        page = 1 if body.get("next") == "page-2" else 0
        return httpx.Response(200, json=catalog[page])

    def client(**options: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **options)

    monkeypatch.setattr(satellite, "httpx", SimpleNamespace(AsyncClient=client))
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
        earth_search_url=SEARCH.removesuffix("/search"),
    )
    monkeypatch.setattr(satellite, "get_settings", lambda: settings)

    stats = satellite.process_vineyard_task(
        vineyard, "2026-03-01T00:00:00+00:00", "2026-10-16T23:59:59+00:00"
    )

    # Повторы одного пролёта отброшены: остались три съёмки
    assert stats == {"found": 3, "processed": 3, "cloudy": 1, "skipped": 0, "failed": 0}
    first, second = requests
    assert first["intersects"]["type"] in ("Polygon", "MultiPolygon")
    assert first["query"] == {"eo:cloud_cover": {"lte": 40}}
    assert first["datetime"].startswith("2026-03-01")
    assert second == {"next": "page-2"}

    history = api_client.get(
        f"/api/v1/vineyards/{vineyard}/vegetation", headers=headers
    ).json()
    assert [item["acquired_at"][:10] for item in history] == [
        "2026-07-10",
        "2026-07-25",
        "2026-08-09",
    ]
    july, cloudy, august = history
    # 10 июля — переобработанная версия: красный 0,15 − 0,1, ближний ИК 0,3 − 0,1
    assert july["ndvi"]["mean"] == pytest.approx((0.2 - 0.05) / 0.25, abs=1e-3)
    assert cloudy["platform"] == "sentinel-2b"
    assert cloudy["ndvi"] is None
    assert cloudy["parcel_cloud_cover"] == 100
    assert august["ndvi"]["mean"] == pytest.approx(0.714, abs=1e-3)
    assert [item["ndvi_map"] is not None for item in history] == [True, False, True]
    # Три чистых снимка — мало для нормы участка: тревог нет
    assert (
        api_client.get(
            f"/api/v1/vineyards/{vineyard}/vegetation/anomalies", headers=headers
        ).json()
        == []
    )

    tile = f"/api/v1/tiles/ndvi/{tile_of(28.8606, 47.1404, 16)}.png"
    assert api_client.get(f"{tile}?day=2026-07-01", headers=headers).status_code == 204
    assert api_client.get(f"{tile}?day=2026-07-20", headers=headers).status_code == 200
    assert api_client.get(tile, headers=headers).status_code == 200
