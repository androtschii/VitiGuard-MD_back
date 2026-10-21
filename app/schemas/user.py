from typing import Literal

from pydantic import Field

from app.models.user import UserRole
from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema


class UserRead(EntityResponse):
    """Пользователь в ответах API. Хэша пароля здесь нет намеренно: поля,
    которых нет в схеме, в ответ не попадают."""

    email: str
    full_name: str | None
    role: UserRole
    is_active: bool


class NotificationSettings(RequestSchema):
    """Настройки уведомлений (pr-084): с какого уровня риска присылать
    предупреждения; null — не присылать."""

    alert_level: Literal["medium", "high"] | None


class NotificationSettingsRead(ResponseSchema):
    alert_level: Literal["medium", "high"] | None


class UserUpdate(RequestSchema):
    """Что пользователь может изменить в своём профиле. Email, роль и активность
    сюда не входят: лишние поля дают 422, а не молча игнорируются."""

    full_name: str = Field(min_length=2, max_length=255)
