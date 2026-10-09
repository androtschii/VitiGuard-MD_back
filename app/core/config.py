from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]

# Ключ подписи для локальной разработки и тестов; значение публично, поэтому в
# staging и production оно запрещено
DEV_SECRET_KEY = "dev-only-insecure-secret-key-do-not-use-in-production"
MIN_SECRET_KEY_LENGTH = 32


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
    # Redis: брокер очереди задач Celery и хранилище их результатов (разные базы Redis)
    celery_broker_url: RedisDsn = RedisDsn("redis://127.0.0.1:6379/0")
    celery_result_backend: RedisDsn = RedisDsn("redis://127.0.0.1:6379/1")
    # Мягкий лимит даёт задаче шанс завершиться сам (SoftTimeLimitExceeded), жёсткий
    # останавливает процесс; значения в секундах
    celery_task_soft_time_limit: int = Field(default=300, gt=0)
    celery_task_time_limit: int = Field(default=330, gt=0)

    # Ключ подписи JWT (HS256): тот, кто его знает, может выписать токен любому
    # пользователю. Задаётся в окружении: python -c "import secrets; print(secrets.token_urlsafe(48))"
    secret_key: SecretStr = SecretStr(DEV_SECRET_KEY)
    # Access-токен живёт недолго: его нельзя отозвать, а украденный перестаёт работать
    # сам. Refresh-токен живёт долго, но хранится в БД и отзывается
    access_token_ttl_minutes: int = Field(default=15, gt=0)
    refresh_token_ttl_days: int = Field(default=30, gt=0)
    # Refresh-cookie уходит только по HTTPS. Отключать — лишь для тестов по http://
    refresh_cookie_secure: bool = True

    # Сброс пароля: ссылка из письма ведёт на страницу фронтенда и действует час
    frontend_url: str = "http://127.0.0.1:5173"
    password_reset_ttl_minutes: int = Field(default=60, gt=0)
    # Почта (SMTP). По умолчанию — Mailpit из compose: письма видны в его
    # веб-интерфейсе и никуда не уходят
    smtp_host: str = "127.0.0.1"
    smtp_port: int = Field(default=1025, gt=0)
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    smtp_starttls: bool = False
    mail_from: str = "VitiGuard MD <no-reply@vitiguard.md>"

    # Open-Meteo: погода без ключа API. Прогноз и архив — разные адреса
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # Таймаут запросов к внешним сервисам, секунды
    external_http_timeout: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def check_secret_key(self) -> Self:
        if self.environment in ("local", "test"):
            return self
        secret = self.secret_key.get_secret_value()
        if secret == DEV_SECRET_KEY or len(secret) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"В окружении {self.environment} нужен собственный secret_key "
                f"не короче {MIN_SECRET_KEY_LENGTH} символов"
            )
        return self

    @model_validator(mode="after")
    def check_time_limits(self) -> Self:
        if self.celery_task_soft_time_limit >= self.celery_task_time_limit:
            raise ValueError(
                "celery_task_soft_time_limit должен быть меньше celery_task_time_limit"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
