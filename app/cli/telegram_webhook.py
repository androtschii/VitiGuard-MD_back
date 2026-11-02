"""Регистрация вебхука Telegram на сервере (pr-100).

    python -m app.cli.telegram_webhook

На сервере бот не опрашивает Telegram (app.cli.telegram_polling — для своей
машины), а получает обновления на POST /api/v1/telegram/webhook публичного
адреса. Адрес берётся из VITIGUARD_TELEGRAM_WEBAPP_URL — сайт и API на одном
домене. Заодно настраиваются меню команд и кнопка «Приложение»."""

import asyncio

import httpx2 as httpx

from app.core.config import Settings, get_settings
from app.integrations.telegram import TelegramClient
from app.services.telegram_bot import configure_bot, https_url


def webhook_url(settings: Settings) -> str:
    base = https_url(settings.telegram_webapp_url)
    if base is None:
        raise SystemExit("VITIGUARD_TELEGRAM_WEBAPP_URL должен быть адресом https://")
    return f"{base}{settings.api_v1_prefix}/telegram/webhook"


async def register(telegram: TelegramClient, settings: Settings) -> str:
    if settings.telegram_webhook_secret is None:
        raise SystemExit("Не задан VITIGUARD_TELEGRAM_WEBHOOK_SECRET")
    url = webhook_url(settings)
    await telegram.set_webhook(url, settings.telegram_webhook_secret.get_secret_value())
    await configure_bot(telegram, settings.telegram_webapp_url)
    return url


def main() -> None:  # pragma: no cover — запрос к настоящему Telegram
    settings = get_settings()
    if settings.telegram_bot_token is None:
        raise SystemExit("Не задан VITIGUARD_TELEGRAM_BOT_TOKEN")

    async def run() -> None:
        async with httpx.AsyncClient(timeout=settings.external_http_timeout) as http:
            url = await register(TelegramClient(http, settings), settings)
            print(f"Вебхук бота @{settings.telegram_bot_username}: {url}", flush=True)

    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
