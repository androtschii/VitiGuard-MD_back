import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from memory_cache import MemoryCache
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.config import Settings
from app.integrations.cache import OSM_VERSION, RedisCache, cached

RunSql = Callable[[str], list[tuple[object, ...]]]


class DownRedis:
    """Redis, до которого не достучаться."""

    async def get(self, _: str) -> None:
        raise RedisConnectionError("refused")

    async def set(self, *_: Any, **__: Any) -> None:
        raise RedisConnectionError("refused")

    async def incr(self, _: str) -> None:
        raise OSError("network is unreachable")


async def test_cached_computes_once_per_version() -> None:
    cache = MemoryCache()
    calls: list[int] = []

    async def compute() -> list[int]:
        calls.append(1)
        return [len(calls)]

    assert await cached(cache, "stats", compute) == [1]
    assert await cached(cache, "stats", compute) == [1]
    await cache.bump(OSM_VERSION)
    assert await cached(cache, "stats", compute) == [2]
    assert len(calls) == 2


async def test_unavailable_redis_does_not_break_requests() -> None:
    cache = RedisCache(DownRedis())  # type: ignore[arg-type]

    async def compute() -> dict[str, int]:
        return {"answer": 42}

    assert await cached(cache, "stats", compute) == {"answer": 42}
    assert await cache.version(OSM_VERSION) == 0
    await cache.bump(OSM_VERSION)
    await cache.set_bytes("tile", b"png", 60)
    assert await cache.get_bytes("tile") is None


def test_cache_has_own_redis_database() -> None:
    settings = Settings(_env_file=None)

    cache = RedisCache.from_settings(settings)

    assert str(settings.cache_url) == "redis://127.0.0.1:6379/2"
    assert cache.client.connection_pool.connection_kwargs["db"] == 2


@pytest.mark.integration
async def test_redis_cache_round_trip(redis_url: str) -> None:
    cache = RedisCache(Redis.from_url(f"{redis_url}/2"))
    try:
        await cache.client.flushdb()
        await cache.set_json("key", {"features": [1, 2]}, ttl=60)

        assert await cache.get_json("key") == {"features": [1, 2]}
        assert await cache.get_json("missing") is None
        assert 0 < await cache.client.ttl("key") <= 60
        # Тайлы — байтами как есть, пустой тайл — пустой строкой, а не «нет ключа»
        await cache.set_bytes("tile", b"\x89PNG", ttl=60)
        await cache.set_bytes("empty", b"", ttl=60)
        assert await cache.get_bytes("tile") == b"\x89PNG"
        assert await cache.get_bytes("empty") == b""
        await cache.bump(OSM_VERSION)
        await cache.bump(OSM_VERSION)
        assert await cache.version(OSM_VERSION) == 2
    finally:
        await cache.close()


def login(api_client: TestClient) -> dict[str, str]:
    credentials = {"email": "grower@example.md", "password": "secret123"}
    api_client.post("/api/v1/auth/register", json={**credentials, "full_name": "Ион"})
    token = api_client.post("/api/v1/auth/login", json=credentials).json()[
        "access_token"
    ]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.integration
def test_osm_layer_is_cached_until_import(
    api_client: TestClient, run_sql: RunSql
) -> None:
    headers = login(api_client)
    url = "/api/v1/osm/vineyards?bbox=28.85,47.13,28.88,47.15"
    assert api_client.get(url, headers=headers).json()["features"] == []
    run_sql(
        "INSERT INTO osm_vineyards (osm_id, geom) VALUES ('way/1', ST_GeomFromText("
        "'MULTIPOLYGON(((28.86 47.14,28.862 47.14,28.862 47.141,28.86 47.14)))', 4326))"
    )
    try:
        # Без импорта слой берётся из кэша
        assert api_client.get(url, headers=headers).json()["features"] == []
        cache: MemoryCache = api_client.app.state.cache  # type: ignore[attr-defined]
        asyncio.run(cache.bump(OSM_VERSION))

        features = api_client.get(url, headers=headers).json()["features"]
        assert [f["properties"]["osm_id"] for f in features] == ["way/1"]
    finally:
        run_sql("TRUNCATE osm_vineyards")
