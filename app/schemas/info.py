from app.core.config import Environment
from app.schemas.base import ResponseSchema


class AppInfo(ResponseSchema):
    """Название, версия и окружение сервиса."""

    name: str
    version: str
    environment: Environment
