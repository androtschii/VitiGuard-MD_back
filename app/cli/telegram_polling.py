"""Telegram-бот без вебхука — для запуска на своей машине (pr-082).

    uv run python -m app.cli.telegram_polling

Вебхуку нужен публичный HTTPS-адрес сервера, а у машины разработчика его нет.
Здесь бот сам забирает обновления у Telegram (getUpdates) и передаёт их тому
же обработчику, что и вебхук; фото уходят в ту же задачу Celery, поэтому нужен
и запущенный воркер. Пока у бота зарегистрирован вебхук, Telegram отвечает
на getUpdates ошибкой: вебхук снимается при запуске."""

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import httpx2 as httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import ExternalServiceError
from app.db.session import create_engine, create_session_factory
from app.integrations.cache import RedisCache
from app.integrations.telegram import TelegramClient
from app.services.telegram_bot import Update, configure_bot, handle_update
from app.worker.celery_app import celery_app

POLL_SECONDS = 30
RETRY_PAUSE = 5.0


async def poll(
    settings: Settings,
    telegram: TelegramClient,
    enqueue: Callable[[str, list[Any]], object],
    rounds: int | None = None,
    pause: float = RETRY_PAUSE,
) -> int:
    """Обрабатывает обновления, пока не остановят (rounds — для тестов).
    Возвращает число обработанных обновлений."""
    engine = create_engine(settings)
    cache = RedisCache.from_settings(settings)
    offset = handled = done = 0
    try:
        while rounds is None or done < rounds:
            done += 1
            try:
                updates = await telegram.get_updates(offset, POLL_SECONDS)
            except ExternalServiceError:
                await asyncio.sleep(pause)
                continue
            for raw in updates:
                offset = max(offset, int(raw.get("update_id", 0)) + 1)
                try:
                    update = Update.model_validate(raw)
                except ValidationError:
                    continue
                # Ответ не ушёл — как и в вебхуке, обновление не повторяется
                async with create_session_factory(engine)() as session:
                    with suppress(ExternalServiceError):
                        await handle_update(
                            update,
                            session,
                            cache,
                            telegram,
                            enqueue,
                            settings.telegram_webapp_url,
                        )
                handled += 1
    finally:
        await cache.close()
        await engine.dispose()
    return handled


def main() -> None:  # pragma: no cover — бесконечный цикл с настоящим Telegram
    settings = get_settings()
    if settings.telegram_bot_token is None:
        raise SystemExit("Не задан VITIGUARD_TELEGRAM_BOT_TOKEN")

    async def run() -> None:
        async with httpx.AsyncClient(timeout=settings.external_http_timeout) as http:
            telegram = TelegramClient(http, settings)
            await telegram.delete_webhook()
            # Меню команд и кнопка «Приложение» у поля ввода
            await configure_bot(telegram, settings.telegram_webapp_url)
            print(
                f"Бот @{settings.telegram_bot_username} слушает обновления", flush=True
            )
            await poll(
                settings,
                telegram,
                lambda name, args: celery_app.send_task(name, args=args),
            )

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover — запуск как скрипта
    main()
