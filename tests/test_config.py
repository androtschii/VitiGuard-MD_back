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
    ],
)
def test_invalid_settings_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
