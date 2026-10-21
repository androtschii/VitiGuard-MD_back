import asyncio
import io
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from memory_cache import MemoryCache
from PIL import Image
from raster_scene import PARCEL, make_scene

from app.integrations.cache import ndvi_tiles_version
from app.integrations.earth_search import EarthScene
from app.services.ndvi_tiles import (
    NdviMapImage,
    latest_maps,
    render_tile,
    tile_bounds,
)

# Участок у Крикова
LON, LAT = 28.8606, 47.1404


def tile_of(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def pixel_of(lon: float, lat: float, z: int, x: int, y: int) -> tuple[int, int]:
    n = 2**z
    column = ((lon + 180) / 360 * n - x) * 256
    row = ((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - y) * 256
    return int(row), int(column)


def square_png(color: tuple[int, int, int, int], size: int = 4) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (size, size), color).save(buffer, "PNG")
    return buffer.getvalue()


def corners(west: float, south: float, east: float, north: float) -> list[list[float]]:
    return [[west, north], [east, north], [east, south], [west, south]]


def decode(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_tile_bounds() -> None:
    west, south, east, north = tile_bounds(0, 0, 0)

    assert (west, east) == (-180, 180)
    assert north == pytest.approx(85.0511, abs=1e-4)
    assert south == pytest.approx(-85.0511, abs=1e-4)
    # Четверть мира в правой нижней четверти
    west, south, east, north = tile_bounds(1, 1, 1)
    assert (west, east, north) == (0, 180, 0)
    assert south == pytest.approx(-85.0511, abs=1e-4)


def test_map_lands_on_its_place_in_tile() -> None:
    z = 16
    x, y = tile_of(LON, LAT, z)
    west, south, east, north = LON - 0.001, LAT - 0.0005, LON + 0.001, LAT + 0.0005

    tile = render_tile(
        z,
        x,
        y,
        [
            NdviMapImage(
                square_png((26, 152, 80, 255)), corners(west, south, east, north)
            )
        ],
    )

    assert tile is not None
    pixels = decode(tile)
    inside = pixels[pixel_of(LON, LAT, z, x, y)]
    assert tuple(inside) == (26, 152, 80, 255)
    outside = pixels[pixel_of(LON, north + 0.001, z, x, y)]
    assert outside[3] == 0


def test_transparent_pixels_do_not_cover_neighbours() -> None:
    z = 16
    x, y = tile_of(LON, LAT, z)
    area = corners(LON - 0.001, LAT - 0.0005, LON + 0.001, LAT + 0.0005)

    tile = render_tile(
        z,
        x,
        y,
        [
            NdviMapImage(square_png((165, 0, 38, 255)), area),
            # Облачный снимок соседа поверх: прозрачные пиксели не стирают первый
            NdviMapImage(square_png((0, 0, 0, 0)), area),
        ],
    )

    assert tile is not None
    assert tuple(decode(tile)[pixel_of(LON, LAT, z, x, y)]) == (165, 0, 38, 255)


def test_tile_without_maps_is_empty() -> None:
    z = 16
    x, y = tile_of(LON, LAT, z)
    far = corners(30.0, 46.0, 30.01, 46.01)

    assert render_tile(z, x, y, [NdviMapImage(square_png((0, 0, 0, 255)), far)]) is None
    assert render_tile(z, x, y, []) is None


RunSql = Callable[[str], list[tuple[object, ...]]]


def login(client: TestClient, email: str) -> dict[str, str]:
    credentials = {"email": email, "password": "secret123"}
    client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = client.post("/api/v1/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TaskCache(MemoryCache):
    """Кэш задачи обработки снимков: тот же, что у API в тесте."""

    async def close(self) -> None:
        return None


@pytest.fixture
def processed(
    api_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migrated_database_url: str,
) -> dict[str, str]:
    """Участок с обработанным синтетическим снимком 29 августа 2026."""
    from app.core.config import Settings
    from app.integrations.cache import RedisCache
    from app.integrations.earth_search import EarthSearchClient
    from app.tasks import satellite

    shared = TaskCache()
    api_client.app.state.cache = shared  # type: ignore[attr-defined]
    monkeypatch.setattr(RedisCache, "from_settings", lambda _: shared)
    headers = login(api_client, "grower@example.md")
    vineyard = api_client.post(
        "/api/v1/vineyards",
        headers=headers,
        json={"name": "Криково", "geometry": PARCEL},
    ).json()["id"]
    scene = make_scene(tmp_path)

    async def fake_search(*_: object, **__: object) -> list[EarthScene]:
        return [scene]

    monkeypatch.setattr(EarthSearchClient, "search", fake_search)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(satellite, "get_settings", lambda: settings)
    satellite.process_vineyard_task(
        vineyard, "2026-03-01T00:00:00+00:00", "2026-10-16T23:59:59+00:00"
    )
    return {**headers, "x-vineyard": vineyard}


def tile_url(z: int = 16, **params: str) -> str:
    x, y = tile_of(LON, LAT, z)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"/api/v1/tiles/ndvi/{z}/{x}/{y}.png" + (f"?{query}" if query else "")


@pytest.mark.integration
def test_tile_shows_latest_map_of_own_vineyards(
    api_client: TestClient, processed: dict[str, str]
) -> None:
    response = api_client.get(tile_url(), headers=processed)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=3600"
    pixels = decode(response.content)
    assert pixels[..., 3].any()
    # NDVI 0,71 — зелёная часть палитры
    red, green, _, _ = pixels[pixels[..., 3] > 0][0]
    assert green > red


@pytest.mark.integration
def test_tiles_are_cached_until_layer_changes(
    api_client: TestClient,
    processed: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api.routes import tiles

    cache: MemoryCache = api_client.app.state.cache  # type: ignore[attr-defined]
    owner = api_client.get("/api/v1/users/me", headers=processed).json()["id"]
    # Обработка снимка уже подняла версию слоя владельца
    assert asyncio.run(cache.version(ndvi_tiles_version(owner))) == 1
    queries: list[int] = []
    reading = latest_maps

    async def counted(*args: Any) -> Any:
        queries.append(1)
        return await reading(*args)

    monkeypatch.setattr(tiles, "latest_maps", counted)

    first = api_client.get(tile_url(), headers=processed)
    second = api_client.get(tile_url(), headers=processed)
    api_client.get(tile_url(day="2026-08-01"), headers=processed)
    empty = api_client.get(tile_url(day="2026-08-01"), headers=processed)

    assert second.content == first.content
    assert empty.status_code == 204
    # По одному запросу к базе на тайл: повторы — из кэша, и пустые тоже
    assert len(queries) == 2

    # Удалённый участок — новая версия, тайл собирается заново и уже пуст
    api_client.delete(f"/api/v1/vineyards/{processed['x-vineyard']}", headers=processed)
    assert api_client.get(tile_url(), headers=processed).status_code == 204
    assert len(queries) == 3


@pytest.mark.integration
def test_empty_tiles_are_204(api_client: TestClient, processed: dict[str, str]) -> None:
    # Снимок 29 августа — на 1 августа картограммы ещё нет
    assert (
        api_client.get(tile_url(day="2026-08-01"), headers=processed).status_code == 204
    )
    assert (
        api_client.get(tile_url(day="2026-08-29"), headers=processed).status_code == 200
    )
    # Мелкий масштаб, тайл за пределами сетки и чужой пользователь
    assert api_client.get(tile_url(z=8), headers=processed).status_code == 204
    assert (
        api_client.get(
            "/api/v1/tiles/ndvi/16/99999/0.png", headers=processed
        ).status_code
        == 204
    )
    stranger = login(api_client, "stranger@example.md")
    assert api_client.get(tile_url(), headers=stranger).status_code == 204
    assert api_client.get(tile_url()).status_code == 401
    assert (
        api_client.get("/api/v1/tiles/ndvi/20/0/0.png", headers=processed).status_code
        == 422
    )
