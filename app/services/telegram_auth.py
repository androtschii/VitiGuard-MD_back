"""Вход в мини-приложение Telegram без пароля (frontend/pr-090).

Telegram передаёт мини-приложению данные запуска (`initData`): кто открыл
приложение, когда, и подпись HMAC-SHA256, сделанную ключом от токена бота.
Подделать подпись без токена нельзя, поэтому сервер, проверив её, знает, что
приложение открыл именно этот пользователь Telegram. Войти так можно только в
учётную запись, к которой этот пользователь сам привязал чат с ботом (pr-082):
в личном чате номер чата совпадает с номером пользователя."""

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qsl

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ServiceUnavailableError, UnauthorizedError
from app.models.user import User
from app.services.auth import AuthSession, issue_tokens

# Данные запуска старше суток не принимаются: перехваченная строка не должна
# открывать вход навсегда
MAX_AGE = timedelta(days=1)
INVALID = "Данные Telegram не прошли проверку"
NOT_LINKED = "Этот Telegram не привязан к учётной записи VitiGuard MD"


def telegram_user_id(
    init_data: str, bot_token: str, now: datetime | None = None
) -> int:
    """Номер пользователя Telegram из проверенных данных запуска."""
    fields = dict(parse_qsl(init_data, keep_blank_values=True))
    received = fields.pop("hash", "")
    # Строка для проверки — все поля, кроме hash, по алфавиту, через перевод строки
    check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, received):
        raise UnauthorizedError(INVALID)
    try:
        signed_at = datetime.fromtimestamp(int(fields["auth_date"]), UTC)
        user_id = int(json.loads(fields["user"])["id"])
    except (KeyError, ValueError, TypeError) as error:
        raise UnauthorizedError(INVALID) from error
    if (now or datetime.now(UTC)) - signed_at > MAX_AGE:
        raise UnauthorizedError("Данные Telegram устарели: откройте приложение заново")
    return user_id


async def telegram_login(
    session: AsyncSession, init_data: str, settings: Settings
) -> AuthSession:
    if settings.telegram_bot_token is None:
        raise ServiceUnavailableError("Telegram-бот не настроен")
    chat_id = telegram_user_id(
        init_data, settings.telegram_bot_token.get_secret_value()
    )
    user = await session.scalar(
        select(User).where(User.telegram_chat_id == chat_id, User.is_active)
    )
    if user is None:
        raise UnauthorizedError(NOT_LINKED)
    tokens = await issue_tokens(session, user, settings)
    await session.commit()
    return tokens
