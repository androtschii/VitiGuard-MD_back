import uuid
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.api import rate_limit
from app.api.deps import get_app_settings, get_current_user
from app.api.rate_limit import Limit, per_address, per_user
from app.api.routes import ai, auth
from app.core.config import Settings
from app.core.errors import register_error_handlers
from app.integrations.cache import RedisCache


def limited_app(cache_url: str, *, enabled: bool | None = True) -> FastAPI:
    """Маленькое приложение с двумя пределами: без базы и без моделей."""
    settings = Settings(_env_file=None, environment="test", rate_limit_enabled=enabled)
    app = FastAPI()
    register_error_handlers(app)
    app.state.cache = RedisCache(
        Redis.from_url(cache_url, socket_timeout=0.5, socket_connect_timeout=0.5)
    )
    app.dependency_overrides[get_app_settings] = lambda: settings
    # Каждое приложение — свои ключи: тесты не мешают друг другу
    suffix = uuid.uuid4().hex
    two = Limit(f"two-{suffix}", 2, 60)
    other = Limit(f"other-{suffix}", 2, 60)
    user = SimpleNamespace(id=uuid.uuid4())
    app.dependency_overrides[get_current_user] = lambda: user

    @app.get("/a", dependencies=[Depends(per_address(two))])
    async def a() -> str:
        return "ok"

    @app.get("/b", dependencies=[Depends(per_address(other))])
    async def b() -> str:
        return "ok"

    @app.get("/me", dependencies=[Depends(per_user(two))])
    async def me() -> str:
        return "ok"

    return app


@pytest.fixture
def redis_cache_url(redis_url: str) -> str:
    return f"{redis_url}/3"


@pytest.fixture
def client(redis_cache_url: str) -> Iterator[TestClient]:
    with TestClient(limited_app(redis_cache_url)) as test_client:
        yield test_client


@pytest.mark.integration
def test_requests_over_the_limit_get_429_with_retry_after(client: TestClient) -> None:
    assert [client.get("/a").status_code for _ in range(2)] == [200, 200]

    response = client.get("/a")

    assert response.status_code == 429
    assert response.json() == {"detail": "Слишком много запросов. Повторите позже"}
    assert 1 <= int(response.headers["Retry-After"]) <= 60


@pytest.mark.integration
def test_limits_are_counted_separately(client: TestClient) -> None:
    for _ in range(3):
        client.get("/a")

    assert client.get("/b").status_code == 200


@pytest.mark.integration
def test_signed_in_users_are_limited_by_account(client: TestClient) -> None:
    statuses = [client.get("/me").status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


@pytest.mark.integration
def test_disabled_limits_never_reject(redis_cache_url: str) -> None:
    with TestClient(limited_app(redis_cache_url, enabled=False)) as client:
        assert {client.get("/a").status_code for _ in range(5)} == {200}
        assert {client.get("/me").status_code for _ in range(5)} == {200}


def test_unavailable_redis_lets_requests_through() -> None:
    # Порт 1 закрыт: Redis «упал». Ограничение — защита, а не условие работы
    with TestClient(limited_app("redis://127.0.0.1:1/3")) as client:
        assert {client.get("/a").status_code for _ in range(5)} == {200}


def test_limits_are_off_in_tests_and_on_elsewhere() -> None:
    assert not Settings(_env_file=None, environment="test").rate_limits_on
    assert Settings(_env_file=None, environment="local").rate_limits_on
    assert Settings(
        _env_file=None, environment="test", rate_limit_enabled=True
    ).rate_limits_on
    assert not Settings(
        _env_file=None, environment="local", rate_limit_enabled=False
    ).rate_limits_on


def test_client_address_without_client() -> None:
    request = SimpleNamespace(client=None)

    assert rate_limit.client_address(request) == "unknown"  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("router", "path", "limit"),
    [
        (auth.router, "/login", rate_limit.LOGIN),
        (auth.router, "/register", rate_limit.REGISTER),
        (auth.router, "/telegram", rate_limit.TELEGRAM_LOGIN),
        (auth.router, "/password-reset", rate_limit.PASSWORD_RESET),
        (ai.router, "/diagnose", rate_limit.DIAGNOSE_LIMIT),
        (ai.router, "/diagnose/jobs", rate_limit.DIAGNOSE_LIMIT),
    ],
)
def test_sensitive_routes_are_limited(
    router: APIRouter, path: str, limit: Limit
) -> None:
    route = next(
        r
        for r in router.routes
        if isinstance(r, APIRoute)
        and r.path.endswith(path)
        and "POST" in (r.methods or set())
    )
    # Предел зашит в замыкание зависимости: ищем его там
    limits = [
        cell.cell_contents
        for dependency in route.dependencies
        for cell in (dependency.dependency.__closure__ or ())  # type: ignore[union-attr]
    ]
    assert limit in limits
