import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from docker.errors import DockerException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.redis import RedisContainer

from app.core.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
POSTGIS_IMAGE = "postgis/postgis:18-3.6"
REDIS_IMAGE = "redis:8-alpine"
DB_INIT_SCRIPT = ROOT / "docker" / "db" / "10_postgis.sh"


@pytest.fixture
def settings() -> Settings:
    # Без локального .env, чтобы результат не зависел от машины разработчика
    return Settings(_env_file=None, environment="test")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """PostgreSQL с PostGIS в Docker на всю тестовую сессию."""
    try:
        container = (
            PostgresContainer(POSTGIS_IMAGE, driver="asyncpg")
            .with_volume_mapping(
                str(DB_INIT_SCRIPT), "/docker-entrypoint-initdb.d/10_postgis.sh"
            )
            .start()
        )
    except DockerException:
        # Без Docker интеграционные тесты локально пропускаются, а в CI должны падать
        if os.getenv("CI"):
            raise
        pytest.skip("Docker недоступен")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """Redis в Docker на всю тестовую сессию (без номера базы в конце)."""
    try:
        container = RedisContainer(REDIS_IMAGE).start()
    except DockerException:
        if os.getenv("CI"):
            raise
        pytest.skip("Docker недоступен")
    try:
        host = container.get_container_host_ip()
        yield f"redis://{host}:{container.get_exposed_port(6379)}"
    finally:
        container.stop()


@pytest.fixture
def alembic_config(database_url: str) -> Config:
    config = Config(ROOT / "alembic.ini")
    # alembic.ini читается через configparser, поэтому % в URL нужно экранировать
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


@pytest.fixture
def migrated_database_url(alembic_config: Config, database_url: str) -> str:
    command.upgrade(alembic_config, "head")
    return database_url


@pytest.fixture
async def session(migrated_database_url: str) -> AsyncIterator[AsyncSession]:
    """Сессия внутри транзакции, которая откатывается после теста."""
    engine = create_async_engine(migrated_database_url)
    async with engine.connect() as connection:
        transaction = await connection.begin()
        async with AsyncSession(
            bind=connection,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        ) as db_session:
            yield db_session
        await transaction.rollback()
    await engine.dispose()
