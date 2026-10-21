from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, SessionDep
from app.models.enums import RiskLevel
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.errors import ErrorResponse
from app.schemas.user import (
    NotificationSettings,
    NotificationSettingsRead,
    UserRead,
    UserUpdate,
)

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
