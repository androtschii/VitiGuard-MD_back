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


def ndvi_tiles_version(owner_id: object) -> str:
    """Версия тайлов NDVI владельца: растёт с новым снимком и с изменением
    или удалением участка."""
    return f"tiles:ndvi:{owner_id}:version"


class Cache(Protocol):
    async def get_json(self, key: str) -> Any | None: ...

    async def set_json(self, key: str, value: Any, ttl: int) -> None: ...

    async def get_bytes(self, key: str) -> bytes | None: ...

    async def set_bytes(self, key: str, value: bytes, ttl: int) -> None: ...

    async def pop_json(self, key: str) -> Any | None: ...

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

    async def get_bytes(self, key: str) -> bytes | None:
        try:
            raw: bytes | None = await self.client.get(key)
        except (RedisError, OSError):
            return None
        return raw

    async def set_bytes(self, key: str, value: bytes, ttl: int) -> None:
        try:
            await self.client.set(key, value, ex=ttl)
        except (RedisError, OSError):
            return

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get_bytes(key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        await self.set_bytes(key, json.dumps(value).encode(), ttl)

    async def pop_json(self, key: str) -> Any | None:
        """Прочитать и удалить одним действием: одноразовые коды не
        срабатывают дважды, даже если два запроса пришли одновременно."""
        try:
            # GET и DEL в одной транзакции MULTI: GETDEL есть только с Redis 6.2
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.get(key)
                pipe.delete(key)
                raw, _ = await pipe.execute()
        except (RedisError, OSError):
            return None
        return None if raw is None else json.loads(raw)

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
