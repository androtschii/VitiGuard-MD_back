from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]


class Settings(BaseSettings):
    """Настройки сервиса из переменных окружения VITIGUARD_* и файла .env."""

    model_config = SettingsConfigDict(
        env_prefix="VITIGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VitiGuard MD API"
    environment: Environment = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://vitiguard:vitiguard@localhost:5432/vitiguard"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
