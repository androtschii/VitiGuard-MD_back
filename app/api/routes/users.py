from fastapi import APIRouter, status

from app.api.deps import CacheDep, CurrentUserDep, SessionDep, SettingsDep
from app.core.errors import ServiceUnavailableError
from app.models.enums import RiskLevel
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.errors import ErrorResponse
from app.schemas.user import (
    NotificationSettings,
    NotificationSettingsRead,
    TelegramLinkRead,
    UserRead,
    UserUpdate,
)
from app.services.telegram_bot import LINK_TTL_SECONDS, create_link_code

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
)


@router.get("/me", summary="Мой профиль")
async def read_me(user: CurrentUserDep) -> UserRead:
    return UserRead.model_validate(user)


@router.patch("/me", summary="Изменение профиля")
async def update_me(
    data: UserUpdate, user: CurrentUserDep, session: SessionDep
) -> UserRead:
    updated = await UserRepository(session).update(user, **data.model_dump())
    await session.commit()
    return UserRead.model_validate(updated)


def to_settings(user: User) -> NotificationSettingsRead:
    level = user.alert_level
    return NotificationSettingsRead(
        alert_level=None if level is None else level.value  # type: ignore[arg-type]
    )


@router.get("/me/notifications", summary="Мои настройки уведомлений")
async def read_notifications(user: CurrentUserDep) -> NotificationSettingsRead:
    return to_settings(user)


@router.put("/me/notifications", summary="Изменение настроек уведомлений")
async def update_notifications(
    data: NotificationSettings, user: CurrentUserDep, session: SessionDep
) -> NotificationSettingsRead:
    user.alert_level = None if data.alert_level is None else RiskLevel(data.alert_level)
    await session.commit()
    await session.refresh(user)
    return to_settings(user)


@router.post(
    "/me/telegram",
    summary="Ссылка для привязки Telegram-бота",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def create_telegram_link(
    user: CurrentUserDep, settings: SettingsDep, cache: CacheDep
) -> TelegramLinkRead:
    if settings.telegram_bot_token is None:
        raise ServiceUnavailableError("Telegram-бот не настроен")
    code = await create_link_code(cache, user.id)
    return TelegramLinkRead(
        url=f"https://t.me/{settings.telegram_bot_username}?start={code}",
        expires_in=LINK_TTL_SECONDS,
    )


@router.delete(
    "/me/telegram",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Отвязать Telegram-бота",
)
async def unlink_telegram(user: CurrentUserDep, session: SessionDep) -> None:
    user.telegram_chat_id = None
    await session.commit()
