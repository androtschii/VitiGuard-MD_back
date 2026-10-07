import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from docker.errors import DockerException
from fastapi.testclient import TestClient
from testcontainers.community.postgres import PostgresContainer

from app.core.config import Settings
from app.main import create_app

POSTGIS_IMAGE = "postgis/postgis:18-3.6"
DB_INIT_SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "db" / "10_postgis.sh"


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
