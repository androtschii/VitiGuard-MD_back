from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
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
        "postgresql+asyncpg://vitiguard:vitiguard@127.0.0.1:5432/vitiguard"
    )
    # Пул соединений: постоянных соединений pool_size, при пиках ещё до max_overflow
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)
    # Сколько секунд запрос ждёт свободное соединение, прежде чем получить ошибку
    database_pool_timeout: float = Field(default=30, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
