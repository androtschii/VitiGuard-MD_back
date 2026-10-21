from typing import Literal

from pydantic import Field, computed_field

from app.models.user import UserRole
from app.schemas.base import EntityResponse, RequestSchema, ResponseSchema


class UserRead(EntityResponse):
    """Пользователь в ответах API. Хэша пароля здесь нет намеренно: поля,
    которых нет в схеме, в ответ не попадают."""

    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    telegram_chat_id: int | None = Field(default=None, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def telegram_linked(self) -> bool:
        """Привязан ли чат Telegram-бота (pr-082); сам номер чата не отдаётся."""
        return self.telegram_chat_id is not None


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


class TelegramLinkRead(ResponseSchema):
    """Одноразовая ссылка на бота: переход по ней привязывает чат."""

    url: str = Field(examples=["https://t.me/VitiGuardMDBot?start=Zk3q9..."])
    expires_in: int = Field(description="Сколько секунд ссылка действует")
