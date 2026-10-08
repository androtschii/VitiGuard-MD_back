from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import AdminDep, AgronomistDep, CurrentUserDep
from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration

RunSql = Callable[[str], list[tuple[object, ...]]]
PASSWORD = "secret123"


@pytest.fixture
def client(migrated_database_url: str, run_sql: RunSql) -> Iterator[TestClient]:
    """Приложение с эндпоинтами для каждого уровня доступа (только для теста)."""
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url=migrated_database_url,  # type: ignore[arg-type]
    )
    app = create_app(settings)

    @app.get("/any")
    async def any_user(user: CurrentUserDep) -> str:
        return user.role

    @app.get("/agronomist")
    async def agronomist_only(user: AgronomistDep) -> str:
        return user.role

    @app.get("/admin")
    async def admin_only(user: AdminDep) -> str:
        return user.role

    with TestClient(app) as test_client:
        yield test_client
    run_sql("TRUNCATE users CASCADE")


def token_for(client: TestClient, run_sql: RunSql, role: str) -> str:
    email = f"{role}@example.md"
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD, "full_name": "Тест"},
    )
    # Администратора через регистрацию не создать — роль выдаётся напрямую в базе
    run_sql(f"UPDATE users SET role = '{role}' WHERE email = '{email}'")
    response = client.post(
        "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
    )
    token: str = response.json()["access_token"]
    return token


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("user", {"/any": 200, "/agronomist": 403, "/admin": 403}),
        ("agronomist", {"/any": 200, "/agronomist": 200, "/admin": 403}),
        ("admin", {"/any": 200, "/agronomist": 200, "/admin": 200}),
    ],
)
def test_access_matrix(
    client: TestClient, run_sql: RunSql, role: str, expected: dict[str, int]
) -> None:
    headers = {"Authorization": f"Bearer {token_for(client, run_sql, role)}"}

    actual = {path: client.get(path, headers=headers).status_code for path in expected}

    assert actual == expected


def test_forbidden_response_explains_the_reason(
    client: TestClient, run_sql: RunSql
) -> None:
    headers = {"Authorization": f"Bearer {token_for(client, run_sql, 'user')}"}

    response = client.get("/admin", headers=headers)

    assert response.json() == {"detail": "Недостаточно прав"}
    # 403 — не про вход: заголовок, предлагающий войти, здесь неуместен
    assert "www-authenticate" not in response.headers


def test_role_checks_require_login_first(client: TestClient) -> None:
    for path in ("/agronomist", "/admin"):
        response = client.get(path)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_demoted_user_loses_access_immediately(
    client: TestClient, run_sql: RunSql
) -> None:
    headers = {"Authorization": f"Bearer {token_for(client, run_sql, 'admin')}"}
    run_sql("UPDATE users SET role = 'user'")

    assert client.get("/admin", headers=headers).status_code == 403
