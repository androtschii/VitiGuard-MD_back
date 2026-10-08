from app.models.user import UserRole
from app.schemas.base import EntityResponse


class UserRead(EntityResponse):
    """Пользователь в ответах API. Хэша пароля здесь нет намеренно: поля,
    которых нет в схеме, в ответ не попадают."""

    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
