from app.core.config import Environment
from app.schemas.base import ResponseSchema


class AppInfo(ResponseSchema):
    """Название, версия и окружение сервиса."""

    name: str
    version: str
    environment: Environment
    # Включён ли демо-вход: страница входа показывает кнопки ролей
    demo_login: bool = False
