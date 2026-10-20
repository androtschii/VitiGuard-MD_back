import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings

# Версия данных OpenStreetMap: импорт увеличивает её, и ключи прежней версии
# больше не читаются, а сами истекают по сроку
OSM_VERSION = "gis:osm:version"
# Слой OSM меняется только при импорте, раз в недели: сутки — с запасом
GIS_TTL_SECONDS = 24 * 3600


class Cache(Protocol):
    async def get_json(self, key: str) -> Any | None: ...

    async def set_json(self, key: str, value: Any, ttl: int) -> None: ...

    async def version(self, name: str) -> int: ...

    async def bump(self, name: str) -> None: ...


class RedisCache:
    """Кэш тяжёлых запросов в Redis. Кэш — ускорение, а не источник данных:
    недоступный Redis не ломает ответ, запрос просто считается заново."""

    def __init__(self, client: Redis) -> None:
        self.client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "RedisCache":
        # Короткие таймауты: без Redis ответ не должен ждать секунды
        return cls(
            Redis.from_url(
                str(settings.cache_url),
                socket_timeout=0.5,
                socket_connect_timeout=0.5,
            )
        )

    async def get_json(self, key: str) -> Any | None:
        try:
            raw = await self.client.get(key)
        except (RedisError, OSError):
            return None
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self.client.set(key, json.dumps(value), ex=ttl)
        except (RedisError, OSError):
            return

    async def version(self, name: str) -> int:
        try:
            raw = await self.client.get(name)
        except (RedisError, OSError):
            return 0
        return int(raw or 0)

    async def bump(self, name: str) -> None:
        try:
            await self.client.incr(name)
        except (RedisError, OSError):
            return

    async def close(self) -> None:
        await self.client.aclose()


async def cached(cache: Cache, key: str, compute: Callable[[], Awaitable[Any]]) -> Any:
    """Значение из кэша по ключу текущей версии данных OSM или посчитанное заново."""
    versioned = f"{key}:v{await cache.version(OSM_VERSION)}"
    value = await cache.get_json(versioned)
    if value is None:
        value = await compute()
        await cache.set_json(versioned, value, GIS_TTL_SECONDS)
    return value
