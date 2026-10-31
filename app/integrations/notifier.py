import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from redis.asyncio import Redis
from redis.asyncio.client import PubSub
from redis.exceptions import RedisError

from app.core.config import Settings


def alerts_channel(user_id: uuid.UUID) -> str:
    return f"alerts:{user_id}"


class Notifier(Protocol):
    async def alerts_changed(self, user_id: uuid.UUID) -> None: ...


class RedisNotifier:
    """Сигнал «предупреждения изменились» через Redis pub/sub (pr-093).

    Предупреждения создаёт и API (кнопка «Пересчитать»), и воркер Celery
    (пересчёт по расписанию), а WebSocket держит процесс API: общий канал
    Redis связывает их. В сообщении нет данных — только повод перечитать
    число непрочитанных из базы, поэтому потерянный сигнал ничего не ломает."""

    def __init__(self, client: Redis, subscriber: Redis) -> None:
        self.client = client
        self.subscriber = subscriber

    @classmethod
    def from_settings(cls, settings: Settings) -> "RedisNotifier":
        # Подписка большую часть времени молчит, поэтому таймаут чтения ей
        # не нужен: с ним она обрывалась бы после первой же паузы
        return cls(
            Redis.from_url(
                str(settings.cache_url),
                socket_timeout=0.5,
                socket_connect_timeout=0.5,
            ),
            Redis.from_url(
                str(settings.cache_url),
                socket_connect_timeout=0.5,
            ),
        )

    async def alerts_changed(self, user_id: uuid.UUID) -> None:
        # Уведомление — удобство: без Redis клиент узнает о предупреждениях
        # при следующем опросе, а пересчёт риска не должен из-за этого падать
        try:
            await self.client.publish(alerts_channel(user_id), "changed")
        except (RedisError, OSError):
            return

    @asynccontextmanager
    async def subscribe(self, user_id: uuid.UUID) -> AsyncIterator[PubSub]:
        pubsub = self.subscriber.pubsub()
        await pubsub.subscribe(alerts_channel(user_id))
        try:
            yield pubsub
        finally:
            await pubsub.unsubscribe()
            # В redis-py у PubSub.aclose нет аннотаций типов
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def close(self) -> None:
        await self.client.aclose()
        await self.subscriber.aclose()
