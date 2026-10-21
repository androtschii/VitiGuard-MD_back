import secrets
from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, Header, status

from app.api.deps import (
    CacheDep,
    SessionDep,
    SettingsDep,
    TaskQueueDep,
    TelegramDep,
)
from app.core.errors import ExternalServiceError, ForbiddenError, NotFoundError
from app.schemas.errors import ErrorResponse
from app.services.telegram_bot import Update, handle_update

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post(
    "/webhook",
    summary="Обновления Telegram-бота (вызывает Telegram)",
    responses={
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
)
async def telegram_webhook(
    update: Update,
    settings: SettingsDep,
    session: SessionDep,
    cache: CacheDep,
    telegram: TelegramDep,
    enqueue: TaskQueueDep,
    secret_token: Annotated[
        str | None, Header(alias="X-Telegram-Bot-Api-Secret-Token")
    ] = None,
) -> dict[str, bool]:
    if telegram is None or settings.telegram_webhook_secret is None:
        raise NotFoundError("Telegram-бот не настроен")
    # Адрес вебхука известен не только Telegram: без секрета обновление не принимается
    expected = settings.telegram_webhook_secret.get_secret_value()
    if not secrets.compare_digest(secret_token or "", expected):
        raise ForbiddenError("Неверный секрет вебхука")
    # Ответ в чат не ушёл — не страшно. Ошибка вебхука заставила бы Telegram
    # повторять то же обновление, и фото ушло бы на анализ дважды
    with suppress(ExternalServiceError):
        await handle_update(update, session, cache, telegram, enqueue)
    return {"ok": True}
