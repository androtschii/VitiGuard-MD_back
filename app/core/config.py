from functools import lru_cache
from pathlib import Path
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
    # Уровень журнала; в production DEBUG выдаёт лишнее и замедляет сервис
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_v1_prefix: str = "/api/v1"
    # Адреса веб-клиента для CORS. Пусто по умолчанию: в разработке клиент ходит
    # через прокси Vite на тот же адрес, и CORS не нужен вовсе
    cors_origins: list[str] = []
    # Ограничение частоты запросов (pr-090). None — включено везде, кроме тестов:
    # иначе обычные тесты упирались бы в предел входа с одного адреса
    rate_limit_enabled: bool | None = None
    # Токен для /metrics; без него метрики открыты (для своей машины)
    metrics_token: SecretStr | None = None
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
    # Кэш тяжёлых ГИС-запросов (слой OSM, статистика районов) — своя база Redis
    cache_url: RedisDsn = RedisDsn("redis://127.0.0.1:6379/2")
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
    # Ансамблевый прогноз для вероятности вспышки (pr-077): ICON-EPS — 40 членов,
    # над Европой с уточнением региональной моделью
    open_meteo_ensemble_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    open_meteo_ensemble_model: str = "icon_seamless"
    # Каталог снимков Copernicus Data Space Ecosystem (STAC API): поиск без учётной записи
    copernicus_stac_url: str = "https://stac.dataspace.copernicus.eu/v1"
    # Те же снимки Sentinel-2 L2A в открытом хранилище AWS: каналы читаются без
    # учётной записи и только в окне участка (pr-059)
    earth_search_url: str = "https://earth-search.aws.element84.com/v1"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    # Выгрузка виноградников всей страны занимает десятки секунд
    overpass_timeout: int = 180
    # Telegram-бот (pr-082, pr-083): без токена бот выключен, вебхук отвечает 404.
    # Секрет вебхука Telegram присылает в заголовке — так чужие POST отсекаются
    telegram_bot_token: SecretStr | None = None
    telegram_bot_username: str = "VitiGuardMDBot"
    telegram_webhook_secret: SecretStr | None = None
    telegram_api_url: str = "https://api.telegram.org"
    # Публичный HTTPS-адрес веб-клиента для кнопок бота: мини-приложение и
    # ссылка для телефона. Telegram принимает в кнопках только https, а
    # 127.0.0.1 с телефона недоступен — без адреса бот кнопок не показывает
    telegram_webapp_url: str | None = None
    # Снимки с облачностью сцены выше порога (в процентах) не берутся
    max_scene_cloud_cover: float = Field(default=10, ge=0, le=100)
    # Для расчёта индексов отсев по облачности сцены мягче: облака над самим
    # участком проверяются по маске SCL, а сцена с 40 % облаков часто чиста над ним
    vegetation_max_scene_cloud_cover: float = Field(default=40, ge=0, le=100)
    # Как часто Celery Beat пересчитывает риск всех участков, часы
    risk_refresh_hours: float = Field(default=6, gt=0)
    # Таймаут запросов к внешним сервисам, секунды
    external_http_timeout: float = Field(default=10, gt=0)
    # Модель диагностики по фото листа (ONNX, ml/pr-010). Рядом лежит файл с тем
    # же именем и расширением .json: название модели, классы и размер входа
    diagnosis_model_path: Path = Path("models/diagnosis.onnx")
    # Сегментация поражения (U-Net, ml/pr-013); без неё диагноз ставится без маски
    lesion_model_path: Path = Path("models/lesion.onnx")
    # S3-совместимое хранилище фото листов (pr-039): RustFS в Docker Compose.
    # Ключи по умолчанию — только для локальной разработки
    s3_endpoint: str = "127.0.0.1:9000"
    s3_access_key: str = "vitiguard"
    s3_secret_key: SecretStr = SecretStr("vitiguard-secret")
    s3_bucket: str = "leaf-images"
    s3_secure: bool = False

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

    @property
    def rate_limits_on(self) -> bool:
        if self.rate_limit_enabled is None:
            return self.environment != "test"
        return self.rate_limit_enabled


@lru_cache
def get_settings() -> Settings:
    return Settings()
