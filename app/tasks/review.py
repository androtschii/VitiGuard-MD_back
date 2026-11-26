import asyncio
import uuid

import httpx2 as httpx
from celery import shared_task

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.integrations.telegram import TelegramClient
from app.services.review_notify import notify_review

NOTIFY_REVIEW = "app.tasks.review.notify_review"


@shared_task(name=NOTIFY_REVIEW)
def notify_review_task(result_id: str) -> dict[str, int]:
    """Уведомление о заключении агронома идёт фоном: заключение сохраняется
    сразу, даже если Telegram или push-сервис сейчас недоступны."""
    settings = get_settings()

    async def run() -> dict[str, int]:
        engine = create_engine(settings)
        try:
            async with (
                httpx.AsyncClient(timeout=30) as http,
                create_session_factory(engine)() as session,
            ):
                telegram = (
                    TelegramClient(http, settings)
                    if settings.telegram_bot_token is not None
                    else None
                )
                return await notify_review(
                    session, settings, telegram, uuid.UUID(result_id)
                )
        finally:
            await engine.dispose()

    return asyncio.run(run())
