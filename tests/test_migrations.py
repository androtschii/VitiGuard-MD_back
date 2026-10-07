import asyncio

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.integration


def extension_version(database_url: str, name: str) -> str | None:
    async def query() -> str | None:
        engine = create_async_engine(database_url)
        try:
            async with engine.connect() as connection:
                version: str | None = await connection.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname = :name"),
                    {"name": name},
                )
                return version
        finally:
            await engine.dispose()

    return asyncio.run(query())


def test_upgrade_downgrade_upgrade(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")


def test_models_match_migrations(alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")
    # Падает, если модели изменились, а миграция под них не создана
    command.check(alembic_config)


def test_postgis_extension(alembic_config: Config, database_url: str) -> None:
    command.upgrade(alembic_config, "head")
    assert extension_version(database_url, "postgis") is not None

    command.downgrade(alembic_config, "base")
    assert extension_version(database_url, "postgis") is None
