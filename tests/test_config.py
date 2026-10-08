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
