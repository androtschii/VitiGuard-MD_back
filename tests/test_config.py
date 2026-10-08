import pytest
from pydantic import ValidationError

from app.core.config import Settings

DATABASE_URL = "postgresql+asyncpg://user:secret@db:5432/vitiguard"


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITIGUARD_ENVIRONMENT", "staging")
    monkeypatch.setenv("VITIGUARD_DEBUG", "true")
    monkeypatch.setenv("VITIGUARD_DATABASE_URL", DATABASE_URL)

    settings = Settings(_env_file=None)

    assert settings.environment == "staging"
    assert settings.debug is True
    assert str(settings.database_url) == DATABASE_URL


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("VITIGUARD_ENVIRONMENT", "prod"),
        ("VITIGUARD_DATABASE_URL", "mysql://user:secret@db:3306/vitiguard"),
        ("VITIGUARD_DATABASE_POOL_SIZE", "0"),
        ("VITIGUARD_DATABASE_MAX_OVERFLOW", "-1"),
        ("VITIGUARD_DATABASE_POOL_TIMEOUT", "0"),
        ("VITIGUARD_CELERY_BROKER_URL", "amqp://guest@rabbit//"),
        ("VITIGUARD_CELERY_RESULT_BACKEND", "http://redis:6379/1"),
        ("VITIGUARD_CELERY_TASK_SOFT_TIME_LIMIT", "0"),
    ],
)
def test_invalid_settings_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_pool_settings_have_defaults_and_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = Settings(_env_file=None)
    assert (defaults.database_pool_size, defaults.database_max_overflow) == (10, 10)
    assert defaults.database_pool_timeout == 30

    monkeypatch.setenv("VITIGUARD_DATABASE_POOL_SIZE", "3")
    monkeypatch.setenv("VITIGUARD_DATABASE_POOL_TIMEOUT", "5.5")

    settings = Settings(_env_file=None)

    assert settings.database_pool_size == 3
    assert settings.database_pool_timeout == 5.5


def test_celery_defaults_use_separate_redis_databases() -> None:
    settings = Settings(_env_file=None)

    assert str(settings.celery_broker_url) == "redis://127.0.0.1:6379/0"
    assert str(settings.celery_result_backend) == "redis://127.0.0.1:6379/1"
    assert settings.celery_task_soft_time_limit < settings.celery_task_time_limit


def test_soft_time_limit_must_be_below_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VITIGUARD_CELERY_TASK_SOFT_TIME_LIMIT", "60")
    monkeypatch.setenv("VITIGUARD_CELERY_TASK_TIME_LIMIT", "60")

    with pytest.raises(ValidationError, match="должен быть меньше"):
        Settings(_env_file=None)
