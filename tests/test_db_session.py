import asyncio
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.deps import SessionDep
from app.core.config import Settings
from app.db.session import create_engine, create_session_factory
from app.main import create_app


def test_engine_uses_pool_settings() -> None:
    settings = Settings(
        _env_file=None,
        database_pool_size=3,
        database_max_overflow=1,
        database_pool_timeout=5,
    )

    engine = create_engine(settings)

    assert engine.url.drivername == "postgresql+asyncpg"
    assert engine.pool.size() == 3  # type: ignore[attr-defined]
    assert engine.pool.timeout() == 5  # type: ignore[attr-defined]


def test_session_factory_keeps_attributes_after_commit() -> None:
    session_factory = create_session_factory(create_engine(Settings(_env_file=None)))

    assert session_factory.kw["expire_on_commit"] is False


def test_app_starts_without_database_and_disposes_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    monkeypatch.setattr("app.main.create_engine", lambda _: engine)

    app = create_app(Settings(_env_file=None, environment="test"))
    with TestClient(app) as client:
        assert client.get("/api/v1/info").status_code == 200
        engine.dispose.assert_not_awaited()

    engine.dispose.assert_awaited_once()


@pytest.fixture
def db_app(migrated_database_url: str) -> FastAPI:
    app = create_app(
        Settings(
            _env_file=None,
            environment="test",
            database_url=migrated_database_url,  # type: ignore[arg-type]
            database_pool_size=2,
        )
    )

    @app.get("/ping")
    async def ping(session: SessionDep) -> int | None:
        value: int | None = await session.scalar(text("SELECT 1"))
        return value

    @app.post("/draft")
    async def draft(session: SessionDep) -> None:
        # Без commit: запись не должна пережить запрос
        await session.execute(
            text("INSERT INTO users (email, hashed_password) VALUES ('a@b.md', 'x')")
        )

    @app.get("/users/count")
    async def count_users(session: SessionDep) -> int | None:
        count: int | None = await session.scalar(text("SELECT count(*) FROM users"))
        return count

    return app


@pytest.fixture
def db_client(db_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(db_app) as client:
        yield client


@pytest.mark.integration
def test_request_gets_working_session(db_client: TestClient) -> None:
    assert db_client.get("/ping").json() == 1


@pytest.mark.integration
def test_uncommitted_changes_are_rolled_back(db_client: TestClient) -> None:
    db_client.post("/draft")

    assert db_client.get("/users/count").json() == 0


@pytest.mark.integration
def test_pool_replaces_connection_closed_by_server(
    db_client: TestClient, migrated_database_url: str
) -> None:
    async def terminate_other_connections() -> None:
        engine = create_async_engine(migrated_database_url)
        async with engine.connect() as connection:
            await connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
                    " WHERE datname = current_database() AND pid <> pg_backend_pid()"
                )
            )
        await engine.dispose()

    assert db_client.get("/ping").json() == 1
    asyncio.run(terminate_other_connections())

    assert db_client.get("/ping").json() == 1
