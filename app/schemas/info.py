from app.core.config import Environment
from app.schemas.base import ResponseSchema


class AppInfo(ResponseSchema):
    name: str
    version: str
    environment: Environment
