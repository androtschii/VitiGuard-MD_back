from pydantic import Field

from app.models.user import UserRole
from app.schemas.base import EntityResponse, RequestSchema


class UserRead(EntityResponse):
    """Пользователь в ответах API. Хэша пароля здесь нет намеренно: поля,
    которых нет в схеме, в ответ не попадают."""

    email: str
    full_name: str | None
    role: UserRole
    is_active: bool


class UserUpdate(RequestSchema):
    """Что пользователь может изменить в своём профиле. Email, роль и активность
    сюда не входят: лишние поля дают 422, а не молча игнорируются."""

    full_name: str = Field(min_length=2, max_length=255)
