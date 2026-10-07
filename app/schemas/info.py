from pydantic import BaseModel

from app.core.config import Environment


class AppInfo(BaseModel):
    name: str
    version: str
    environment: Environment
