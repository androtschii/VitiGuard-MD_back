from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

pytestmark = pytest.mark.integration

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture
def alembic_config(database_url: str) -> Config:
    config = Config(ALEMBIC_INI)
    # alembic.ini читается через configparser, поэтому % в URL нужно экранировать
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def test_upgrade_downgrade_upgrade(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")


def test_models_match_migrations(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    # Падает, если модели изменились, а миграция под них не создана
    command.check(alembic_config)
